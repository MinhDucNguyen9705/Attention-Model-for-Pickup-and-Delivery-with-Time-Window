import torch
import numpy as np
from torch import nn
import math
from typing import NamedTuple
from torch.utils.checkpoint import checkpoint

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            num_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // num_heads
        if key_dim is None:
            key_dim = val_dim

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_value = nn.Parameter(torch.Tensor(num_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(num_heads, val_dim, embed_dim))

        self.init_parameters()
    
    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, x.size(-1) // self.num_heads)
        return x.permute(2, 0, 1, 3)  # (num_heads, batch_size, seq_len, dim_per_head)

    def forward(self, q, h=None, mask=None):

        if h is None:
            h = q
        
        batch_size, graph_size, input_dim = h.shape
        n_query = q.shape[1]

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        Q = torch.matmul(qflat, self.W_query)
        K = torch.matmul(hflat, self.W_key)
        V = torch.matmul(hflat, self.W_value)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.num_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out

class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(embed_dim, affine=True)
        elif normalization == 'layer':
            self.norm = nn.InstanceNorm1d(embed_dim, affine=True)
        else:
            raise ValueError("Unsupported normalization type: {}".format(normalization))

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if isinstance(self.norm, nn.BatchNorm1d):
            return self.norm(x.view(-1, x.shape[-1])).view(*x.shape)
        elif isinstance(self.norm, nn.InstanceNorm1d):
            return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            raise ValueError("Unsupported normalization type: {}".format(type(self.norm)))
        
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForward, self).__init__()

        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        
    def forward(self, x):

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x

class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        ff_dim=512,
        normalization='batch'
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.ff = FeedForward(embed_dim=embed_dim, ff_dim=ff_dim)

        self.norm1 = Normalization(embed_dim=embed_dim, normalization=normalization)
        self.norm2 = Normalization(embed_dim=embed_dim, normalization=normalization)
        
        # self.act1 = nn.Tanh()
        # self.act2 = nn.Tanh()
    
    def forward(self, x, mask=None):

        mha_out = self.mha(x, h=x, mask=mask)
        x = x + mha_out
        x = self.norm1(x)
        # x = self.act1(x)

        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)
        # x = self.act2(x)

        return x

class GraphAttentionEncoder(nn.Module):

    def __init__(self,
                 num_heads,
                 embed_dim,
                 num_layers,
                 node_dim=None,
                 normalization='batch',
                 ff_dim=512):
        super().__init__()

        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(
                num_heads=num_heads,
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                normalization=normalization
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):

        h = self.init_embed(x) if self.init_embed is not None else x
        for layer in self.layers:
            h = layer(h, mask=mask)
        
        return h, h.mean(dim=1)

class AttentionModelFixed(NamedTuple):
    
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )

class AttentionModel(nn.Module):

    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.tanh_clipping = tanh_clipping
        self.decode_type = None
        self.temp = 1.0
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.problem = problem

        step_context_dim = embed_dim + 1
        node_dim = 3

        self.init_embed_depot = nn.Linear(2, embed_dim)
        self.init_embed = nn.Linear(node_dim, embed_dim)

        self.embedder = GraphAttentionEncoder(
            num_heads=n_heads,
            embed_dim=embed_dim,
            num_layers=n_encode_layers,
            # node_dim=node_dim,
            normalization=normalization
        )

        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embed_dim, bias=False)

        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp
        
    def forward(self, input, return_pi=False):

        if self.checkpoint_encoder and self.training:
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))
        
        _log_p, pi = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, pi)

        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi
        else:
            return cost, ll
    
    def _calc_log_likelihood(self, _log_p, a, mask):

        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            log_p[mask] = 0
        
        return log_p.sum(1)

    def _init_embed(self, input):

        return torch.cat(
            (
                self.init_embed_depot(input['coords'][:, :1, 0:2]),
                self.init_embed(torch.cat(
                    (
                        input['coords'][:, 1:, :],
                        input['demand'][:, 1:, None],
                    ),
                    dim=-1
                ))
            ),
            dim=1
        )
    
    def _inner(self, input, embeddings):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        fixed = self._precompute(embeddings)

        batch_size = embeddings.size(0)

        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]

                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    state = state[unfinished]
                    fixed = fixed[unfinished]
            
            log_p, mask = self._get_log_p(fixed, state)

            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])

            state = state.update(selected)

            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_
            
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1
        
        return torch.stack(outputs, 1), torch.stack(sequences, 1)
    
    def _select_node(self, probs, mask):

        if self.decode_type == "greedy":
            _, selected = probs.max(1)

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        return selected

    def _precompute(self, embeddings, num_steps=1):

        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
    
    def _get_parallel_step_context(self, embeddings, state, from_depot=False):

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if from_depot:
            return torch.cat(
                (
                    embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                    self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                ),
                dim=-1
            )
        else:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous().view(batch_size, num_steps, 1).expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]
                ),
                -1
            )

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):

        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        mask = state.get_mask()

        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask
    
    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

