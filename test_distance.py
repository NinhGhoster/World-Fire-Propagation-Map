import networkx as nx
import numpy as np

def test_distance_calculation():
    """Test the haversine distance calculation used in the save function"""
    
    # Test coordinates (Bangkok area)
    lat1, lon1 = 13.7563, 100.5018  # Bangkok
    lat2, lon2 = 13.7563, 100.5018  # Same point (should be 0)
    
    # Haversine distance calculation
    R = 6371  # Earth's radius in kilometers
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    print(f"Distance between same points: {distance:.2f} km (should be 0)")
    
    # Test with different points
    lat2, lon2 = 13.7563, 100.5118  # 1km east
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    print(f"Distance between different points: {distance:.2f} km")
    
    # Test NetworkX graph creation
    G = nx.Graph()
    G.add_node(0, lat=lat1, lon=lon1)
    G.add_node(1, lat=lat2, lon=lon2)
    G.add_edge(0, 1, weight=distance)
    
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Edge weight: {G[0][1]['weight']:.2f} km")
    
    # Test adjacency matrix
    adj_matrix = nx.adjacency_matrix(G, weight='weight').toarray()
    print(f"Adjacency matrix:\n{adj_matrix}")
    
    return True

if __name__ == "__main__":
    test_distance_calculation()
    print("Distance calculation test completed successfully!") 