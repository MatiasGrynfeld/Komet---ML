from pathlib import Path
import json

data_folder = Path('./raw_data/')
output_file = Path('./data_earthquakes.json')
if output_file.exists():
    output_file.unlink()

def parse_feature(feature):
    parsed_feature = {}
    avoid_keys = ["id", "url", "detail", "code", "ids", "sources", "types", "title", "status", "type"]
    
    if 'properties' in feature:
        for key, value in feature['properties'].items():
            if key not in avoid_keys:
                parsed_feature[key] = value
    
    if 'geometry' in feature:
        for key, value in feature['geometry'].items():
            if key not in avoid_keys:
                parsed_feature[key] = value
    
    return parsed_feature

def join_data_files():
    total_events = 0
    
    files = list(data_folder.glob('earthquake_data_*.json'))
    print(f"Found {len(files)} data files to process")
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('[\n')
        first_feature = True
        
        for i, file in enumerate(files):
            print(f"Processing file {i+1}/{len(files)}: {file.name}...")
            
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if 'features' in data and data['features']:
                        features = data['features']
                        file_event_count = len(features)
                        
                        for j, feature in enumerate(features):
                            parsed_feature = parse_feature(feature)
                            
                            if not first_feature:
                                out_f.write(',\n')
                            else:
                                first_feature = False
                            
                            json.dump(parsed_feature, out_f, separators=(',', ':'))
                            
                            features[j] = None
                        
                        total_events += file_event_count
                        print(f"  ✓ Added {file_event_count} parsed events (total: {total_events:,})")
                        
                        del features
                        del data
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  ❌ Error processing {file.name}: {e}")
                continue
            
            if (i + 1) % 10 == 0:
                import gc
                gc.collect()
        
        out_f.write('\n]')
    
    print(f"\nSuccessfully joined and parsed {total_events:,} events into {output_file}")
    import gc
    gc.collect()

if __name__ == "__main__":
    join_data_files()