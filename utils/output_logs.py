import os
import re
import pandas as pd
from torchinfo import summary

def log_to_excel(results, file_path, output_dir):
    instance_name, valid, msg, n_routes, cost, mean_cap, std_cap, mean_wait, std_wait, routes = results
    
    excel_filename = f"{instance_name}.xlsx"
    output_path = os.path.join(output_dir, excel_filename)

    metadata_list = []
    input_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        clean_text = re.sub(r'\\', '', raw_text)
        tokens = clean_text.split()
        iterator = iter(tokens)
        
        current_section = "METADATA" 
        
        while True:
            try:
                token = next(iterator)
            except StopIteration:
                break

            if token == "NODES":
                current_section = "NODES"
                continue
            elif token == "EDGES":
                current_section = "EDGES" 
                break
            elif token == "EOF":
                break

            if current_section == "METADATA":
                if token.endswith(":"):
                    key = token.strip(":")
                    try:
                        value = next(iterator)
                        metadata_list.append({"Parameter": key, "Value": value})
                    except StopIteration:
                        break

            elif current_section == "NODES":
                try:
                    node_id = int(token)
                    lat = float(next(iterator))
                    lon = float(next(iterator))
                    demand = int(next(iterator))
                    start_tw = int(next(iterator))
                    end_tw = int(next(iterator))
                    service = int(next(iterator))
                    pickup_id = int(next(iterator))
                    delivery_id = int(next(iterator))
                    
                    input_data.append({
                        "ID": node_id,
                        "Lat": lat,
                        "Lon": lon,
                        "Demand": demand,
                        "Start_Time": start_tw,
                        "End_Time": end_tw,
                        "Service_Time": service,
                        "Pickup_ID": pickup_id,
                        "Delivery_ID": delivery_id
                    })
                except (StopIteration, ValueError):
                    continue 

    except Exception as e:
        print(f"Error parsing input file {file_path}: {e}")
        return

    df_meta = pd.DataFrame(metadata_list)
    df_input = pd.DataFrame(input_data)

    summary_data = {
        "Metric": [
            "Instance Name", "Status", "Message", "Total Routes", "Total Cost", 
            "Mean Capacity Used", "Std Dev Capacity", "Mean Wait Time", "Std Dev Wait Time"
        ],
        "Value": [
            instance_name,
            "Valid" if valid else "Invalid",
            msg,
            n_routes,
            cost,
            f"{mean_cap:.2%}",
            f"{std_cap:.2%}",
            f"{mean_wait:.2f}",
            f"{std_wait:.2f}"
        ]
    }
    df_summary = pd.DataFrame(summary_data)

    route_data = []
    for idx, r in enumerate(routes):
        route_str = " -> ".join(map(str, r))
        route_data.append({"Vehicle ID": idx + 1, "Route": route_str})
    df_routes = pd.DataFrame(route_data)

    try:
        os.makedirs(output_dir, exist_ok=True)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            if not df_meta.empty:
                df_meta.to_excel(writer, sheet_name='Input', index=False, startrow=0)
                start_row_nodes = len(df_meta) + 3
            else:
                start_row_nodes = 0
            
            if not df_input.empty:
                df_input.to_excel(writer, sheet_name='Input', index=False, startrow=start_row_nodes)
            else:
                pd.DataFrame(["Error parsing nodes"]).to_excel(writer, sheet_name='Input', index=False, startrow=start_row_nodes)

            df_summary.to_excel(writer, sheet_name='Output', index=False, startrow=0)
            
            start_row_routes = len(df_summary) + 3
            df_routes.to_excel(writer, sheet_name='Output', index=False, startrow=start_row_routes)
            
        print(f"Log saved to: {output_path}")

    except Exception as e:
        print(f"Error writing excel for {instance_name}: {e}")

def log_results(results, output_path):
    with open(output_path, "a", encoding='utf-8') as f:
        f.write("Instance: %s\n" % results[0])
        f.write("Result: %s\n" % ("Valid" if results[1] else "Invalid"))
        f.write("Message: %s\n" % results[2])
        
        if results[1]:
            f.write("Number of Routes: %d\n" % results[3])
            f.write("Total Cost: %d\n" % results[4])
            f.write("Mean Capacity Used: %.2f%%\n" % (results[5] * 100))
            f.write("Std Dev Capacity Used: %.2f%%\n" % (results[6] * 100))
            f.write("Mean Wait Time: %.2f\n" % results[7])
            f.write("Std Dev Wait Time: %.2f\n" % results[8])
            
            # New section to log individual routes
            f.write("Routes Details:\n")
            for i, route in enumerate(results[9]):
                # Format route as 0 -> 5 -> 10 -> 0
                route_str = " -> ".join(map(str, route))
                f.write("  Vehicle %d: %s\n" % (i + 1, route_str))
                
        f.write("\n")

def log_models(model, opts, output_path, training=True):
    with open(output_path, 'w', encoding='utf-8') as f:
        model_stats = summary(model, verbose=0)
        f.write('Model Parameter configs:\n')
        f.write(f'Embedding dim: {opts.embedding_dim}\n')
        f.write(f'Number of encode layers: {opts.n_encode_layers}\n')
        f.write(f'Tanh clipping: {opts.tanh_clipping}\n')
        f.write(f'Normalization: {opts.normalization}\n')
        f.write(f'Number of heads: {opts.n_heads}\n')
        if training:
            f.write('Training configs:\n')
            f.write(f'Number of epochs: {opts.n_epochs}\n')
            f.write(f'Learning rate: {opts.lr_model}\n')
        f.write('\n')
        f.write('Model summary:\n')
        f.write(str(model_stats)+'\n'+'\n')