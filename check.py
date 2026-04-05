import osmnx as ox
import networkx as nx

place_name = "Bangalore, India"
G = ox.graph_from_place(place_name, network_type='drive')
ox.plot_graph(G)
orig = list(G.nodes())[0]
dest = list(G.nodes())[100]
route = nx.shortest_path(G, orig, dest, weight='length')
route = nx.shortest_path(G, orig, dest, weight='length')
ox.plot_graph_route(G, route)