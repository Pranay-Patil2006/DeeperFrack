import re

def test_regex():
    name = "System.ws.Tile[2..17].PEInputBuffer"
    
    # User's suggestion: handle arbitrary [start..end]
    instances = 1
    for match in re.finditer(r'\[(\d+)\.\.(\d+)\]', name):
        start = int(match.group(1))
        end = int(match.group(2))
        instances *= (end - start + 1)
        
    print(f"Instances for {name}: {instances}")

test_regex()
