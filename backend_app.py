"""
PAI Urban Agent — Backend API (Flask)
Run: python app.py  →  http://localhost:5000
"""
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import heapq, random, math, time
from collections import deque, defaultdict
import networkx as nx
import osmnx as ox

app = Flask(__name__)
CORS(app)

# ── Build city graph once on startup ─────────────────────────────────────────
random.seed(42)
ZONE_TYPES = ['residential', 'commercial', 'industrial', 'medical', 'park']

def build_city():
    # MG Road, Bangalore coordinates: (12.9716, 77.5946)
    # Fetch a driving network within 1.5km
    G_osm = ox.graph_from_point((12.9716, 77.5946), dist=1500, network_type='drive')
    
    # Get largest strongly connected component so paths always exist safely
    comps = sorted(nx.strongly_connected_components(G_osm), key=len, reverse=True)
    G_osm = G_osm.subgraph(comps[0]).copy()
    
    G = nx.DiGraph()
    
    # Map OSmnx graph IDs to simpler integer IDs 0, 1, 2...
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(G_osm.nodes())}
    
    # Get min/max lat lon for scaling to 1-9 grid to match frontend expectations
    xs = [data['x'] for n, data in G_osm.nodes(data=True)]
    ys = [data['y'] for n, data in G_osm.nodes(data=True)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    diff_x = max_x - min_x + 1e-9
    diff_y = max_y - min_y + 1e-9
    max_diff = max(diff_x, diff_y)

    for old_id, data in G_osm.nodes(data=True):
        new_id = node_mapping[old_id]
        # scale uniformly to max 8 range to preserve aspect ratio
        scale_x = ((data['x'] - min_x) / max_diff * 8) + 1
        scale_y = ((data['y'] - min_y) / max_diff * 8) + 1
        
        zone = random.choice(ZONE_TYPES)
        pop  = random.randint(200, 5000)
        G.add_node(new_id, pos=(round(scale_x, 4), round(scale_y, 4)), zone=zone, population=pop)

    for u, v, k, data in G_osm.edges(keys=True, data=True):
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        # Use length (in meters) as weight, scaled down
        w = round(data.get('length', 10.0) / 100.0, 2)
        c = round(random.uniform(1.0, 2.8), 2)
        
        if G.has_edge(new_u, new_v):
            if w < G[new_u][new_v]['weight']:
                G[new_u][new_v]['weight'] = w
        else:
            G.add_edge(new_u, new_v, weight=w, congestion=c)
            
    return G

G = build_city()

def graph_to_json():
    nodes = [{"id": n,
              "x": G.nodes[n]['pos'][0],
              "y": G.nodes[n]['pos'][1],
              "zone": G.nodes[n]['zone'],
              "population": G.nodes[n]['population']}
             for n in G.nodes]
    edges = [{"source": u, "target": v,
              "weight": round(d['weight'],2),
              "congestion": round(d['congestion'],2)}
             for u,v,d in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges}

def heuristic(u, goal):
    xu,yu = G.nodes[u]['pos']
    xg,yg = G.nodes[goal]['pos']
    return math.hypot(xu-xg, yu-yg)

def node_dist(u, v):
    xu,yu = G.nodes[u]['pos']
    xv,yv = G.nodes[v]['pos']
    return math.hypot(xu-xv, yu-yv)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def serve_frontend():
    return send_file('frontend_index.html')

@app.route('/api/graph')
def get_graph():
    return jsonify(graph_to_json())

@app.route('/api/search', methods=['POST'])
def search():
    data    = request.json
    algo    = data.get('algorithm','astar')
    start   = int(data.get('start', 0))
    goal    = int(data.get('goal', 15))
    t0      = time.perf_counter()

    if algo == 'bfs':
        path, cost, expanded, steps = _bfs(start, goal)
    elif algo == 'ucs':
        path, cost, expanded, steps = _ucs(start, goal)
    else:
        path, cost, expanded, steps = _astar(start, goal)

    elapsed = round((time.perf_counter()-t0)*1000, 4)
    explanation = _explain_routing(algo, start, goal, path, cost, expanded, elapsed)
    return jsonify({"path": path, "cost": round(cost,3),
                    "expanded": expanded, "time_ms": elapsed,
                    "steps": steps, "explanation": explanation})

@app.route('/api/csp', methods=['POST'])
def csp():
    data = request.json
    k    = int(data.get('k', 3))
    sep  = float(data.get('min_separation', 2.5))
    result = _csp(k, sep)
    return jsonify(result)

@app.route('/api/adversarial', methods=['POST'])
def adversarial():
    data  = request.json
    start = int(data.get('start',0))
    goal  = int(data.get('goal',10))
    result = _minimax_demo(start, goal)
    return jsonify(result)

@app.route('/api/rl', methods=['POST'])
def rl():
    data     = request.json
    goal     = int(data.get('goal', 15))
    episodes = int(data.get('episodes', 400))
    result   = _qlearn(goal, episodes)
    return jsonify(result)

# ── Algorithm implementations ─────────────────────────────────────────────────

def _bfs(start, goal):
    parent = {start: None}; queue = deque([start]); expanded = []
    steps = [{"action": "enqueue", "node": start}]
    while queue:
        node = queue.popleft(); expanded.append(node)
        steps.append({"action": "expand", "node": node})
        if node == goal: break
        for nbr in G.successors(node):
            if nbr not in parent:
                parent[nbr] = node; queue.append(nbr)
    path=[]; n=goal
    while n is not None: path.append(n); n=parent.get(n)
    path.reverse()
    cost = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)) if len(path)>1 else 0
    return path, cost, len(expanded), steps[:40]

def _ucs(start, goal):
    pq=[(0,start,[start])]; visited={}; expanded=[]; steps=[]
    while pq:
        cost,node,path = heapq.heappop(pq)
        if node in visited: continue
        visited[node]=cost; expanded.append(node)
        steps.append({"action":"expand","node":node,"g":round(cost,2)})
        if node==goal: return path,cost,len(expanded),steps[:40]
        for nbr,d in G[node].items():
            if nbr not in visited:
                heapq.heappush(pq,(cost+d['weight'],nbr,path+[nbr]))
    return [],float('inf'),len(expanded),steps[:40]

def _astar(start, goal):
    pq=[(heuristic(start,goal),0,start,[start])]; visited={}; expanded=[]; steps=[]
    while pq:
        f,g,node,path = heapq.heappop(pq)
        if node in visited: continue
        visited[node]=g; expanded.append(node)
        steps.append({"action":"expand","node":node,"g":round(g,2),"h":round(heuristic(node,goal),2)})
        if node==goal: return path,g,len(expanded),steps[:40]
        for nbr,d in G[node].items():
            if nbr not in visited:
                ng=g+d['weight']
                heapq.heappush(pq,(ng+heuristic(nbr,goal),ng,nbr,path+[nbr]))
    return [],float('inf'),len(expanded),steps[:40]

def _explain_routing(algo, start, goal, path, cost, expanded, elapsed):
    algo_map={'bfs':'BFS','ucs':'UCS','astar':'A*'}
    name=algo_map.get(algo,algo)
    lines=[
        f"Algorithm: {name}",
        f"Route: {' → '.join(map(str,path))} ({len(path)-1} hops)",
        f"Total travel time: {round(cost,3)} units",
        f"Nodes expanded: {expanded}  |  Time: {elapsed}ms",
    ]
    if algo=='astar':
        lines.append("Strategy: Used straight-line (Euclidean) heuristic to guide expansion toward goal. Guaranteed optimal because heuristic is admissible (never overestimates).")
    elif algo=='ucs':
        lines.append("Strategy: Expanded nodes in order of cumulative g-cost (Dijkstra). Guaranteed optimal for non-negative weights.")
    else:
        lines.append("Strategy: Level-by-level BFS. Minimises hop count, NOT travel time — path may be sub-optimal by cost.")
    return lines

def _csp(k=3, min_sep=2.5):
    candidates=[n for n in G.nodes if G.nodes[n]['zone'] in ('medical','commercial','residential')]
    if len(candidates) > 30:
        candidates = random.sample(candidates, 30)
    best={'placed':[],'score':-1,'tried':0}

    def coverage(placed):
        covered=set()
        for f in placed:
            for n in G.nodes:
                if node_dist(f,n)<=3.0: covered.add(n)
        return sum(G.nodes[n]['population'] for n in covered)

    def bt(idx,placed):
        best['tried']+=1
        if len(placed)==k:
            s=coverage(placed)
            if s>best['score']:
                best['score']=s; best['placed']=placed[:]
            return
        if idx>=len(candidates): return
        if len(candidates)-idx<k-len(placed): return
        node=candidates[idx]
        if all(node_dist(node,p)>=min_sep for p in placed):
            placed.append(node); bt(idx+1,placed); placed.pop()
        bt(idx+1,placed)

    bt(0,[])
    exp=[]
    for i,f in enumerate(best['placed']):
        nbrs=[n for n in G.nodes if node_dist(f,n)<=3.0]
        pop=sum(G.nodes[n]['population'] for n in nbrs)
        exp.append({"node":f,"zone":G.nodes[f]['zone'],"covers":len(nbrs),"population":pop})
    return {"placed":best['placed'],"score":best['score'],
            "tried":best['tried'],"explanation":exp}

def _minimax_demo(start,goal):
    sub_nodes=[start]+list(nx.single_source_shortest_path(G,start,cutoff=2).keys())[:7]
    sub_nodes=list(set(sub_nodes))[:8]
    Gsub=G.subgraph(sub_nodes).copy()
    paths=list(nx.all_simple_paths(Gsub,start,goal,cutoff=4)) if goal in sub_nodes else []
    if not paths:
        # fallback: use UCS path
        path,cost,_,_ = _ucs(start,goal)
        return {"path":path,"value":round(cost,2),"log":["No adversarial paths found; fallback to UCS"]}
    
    log=[]
    def mm(g,ps,depth,is_max,alpha,beta):
        if depth==0 or not ps:
            costs=[sum(g[ps[i][j]][ps[i][j+1]]['weight'] for j in range(len(ps[i])-1)) for i in range(len(ps))]
            v=min(costs) if costs else 0
            log.append(f"Leaf: min_cost={round(v,2)}")
            return v,None
        if is_max:
            bv,bp=float('inf'),None
            for p in ps:
                c=sum(g[p[j]][p[j+1]]['weight'] for j in range(len(p)-1))
                log.append(f"MAX path {p} cost={round(c,2)}")
                if c<bv: bv,bp=c,p
                beta=min(beta,bv)
                if alpha>=beta: log.append("  α-β prune"); break
            return bv,bp
        else:
            wv,we=float('-inf'),None
            for u,v2 in list(g.edges())[:5]:
                ow=g[u][v2]['weight']
                g[u][v2]['weight']=ow*2.2
                val,_=mm(g,ps,depth-1,True,alpha,beta)
                g[u][v2]['weight']=ow
                log.append(f"MIN sabotage ({u}→{v2}) → val={round(val,2)}")
                if val>wv: wv,we=val,(u,v2)
                alpha=max(alpha,wv)
                if alpha>=beta: break
            return wv,we

    val,bp=mm(Gsub,paths,2,True,float('-inf'),float('inf'))
    return {"path":bp or [],"value":round(val,2),"log":log[:15]}

def _qlearn(goal=15,episodes=400):
    Q=defaultdict(lambda: defaultdict(float))
    alpha,gamma,epsilon=0.3,0.9,0.25
    rewards=[]
    nodes=list(G.nodes)
    Glocal=G.copy()
    for ep in range(episodes):
        s=random.choice([n for n in nodes if n!=goal])
        total=0
        if ep%60==0:
            for u,v,d in Glocal.edges(data=True):
                d['weight']=max(0.5,d['weight']*random.uniform(0.85,1.15))
        for _ in range(25):
            nbrs=list(Glocal.successors(s))
            if not nbrs: break
            a=random.choice(nbrs) if random.random()<epsilon else min(nbrs,key=lambda x:Q[s][x])
            r=-Glocal[s][a]['weight']+(50 if a==goal else 0)
            nbrs2=list(Glocal.successors(a))
            future=min((Q[a][n] for n in nbrs2),default=0)
            Q[s][a]+=alpha*(r+gamma*future-Q[s][a])
            total+=r; s=a
            if s==goal: break
        rewards.append(round(total,2))
    # greedy path
    s=0; path=[s]; visited={s}
    for _ in range(20):
        if s==goal: break
        nbrs=[n for n in G.successors(s) if n not in visited]
        if not nbrs: break
        a=min(nbrs,key=lambda x:Q[s][x])
        path.append(a); visited.add(a); s=a
    # smooth rewards for chart
    w=20; smooth=[]
    for i in range(len(rewards)):
        smooth.append(round(sum(rewards[max(0,i-w):i+1])/min(i+1,w),2))
    return {"path":path,"rewards":rewards[::5],"smoothed":smooth[::5],"episodes":episodes}

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
