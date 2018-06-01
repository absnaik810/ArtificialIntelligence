#!/usr/bin/env python
'''
This is a classical route search problem and following is the abstarction model for the search
	Initial State: The start city provided as an input by user (Source city)
	Goal State: The destination city provided as an input by user
	Successor function: Depends on the routing algorithm given by user,
					For bfs and dfs the neighbour cities of a city (connected by road segments
						For uniform and astar, neighbour with least cost will be picked as a priority city.
	Cost function: Depends on input specified by user.
			   Segments - Minimum number of road segments to traverse to reach goal state
			   Distance - Minimum distance (in miles) to reach destination city
			       Time - Minimum time (in hours) required to reach destination city

Input to the program is source_city, destination_city, routing_algorithm and cost_function to be accepted by user.
Data to the program is taken from city-gps.txt and road-segments.txt contaning the coordinates of various cities and
edges connecting between them respectively.

(1) Which search algorithm seems to work best for each routing options?
	1. Segments.
	   BFS, Uniform and Astar returns optimal solution for segments, but bfs gives the result fast. BFS expands all
	   nodes in level wise manner, hence at some point of expansion, goal node is encountered reutning an immediate in
	   optimal solution. Uniform and astar are slow in execution since, both the algorithms check destination node when
	   an element is next to be expanded opposing to the implementation of BFS. And also, execution time of astar depends
	   on the heursitic function.
	2. Distance.
	   Uniform routing returns the optimal solution. The implementation used here is a greddy technique, for given source
	   node in the graph, the algorithm finds shortest path between that node and every other node. But the running time is
	   a trade-off as a local minimum cost is searched for every node, the algorithm will be very slow. Comparatively, astar
	   routing algorithm returns a near optimal solution in less time due to heuristic function.
	3. Time.
	   Similar to Distance routing option, Uniform routing returns the optimal solution. As said above, uniform routing
	   algorithm returns an optimal solution with trade-off in time. Also, compared to other routing algorithms, astar returns
	   a near optimal solution in less time because of the herusitic function implemented.

(2) Which algorithm is fastest in terms of the amount of computation time required by your program, and by how much, according to your experiments?
	For segments routing option BFS seems to run faster, but astar is fastest for distance and time.
	Below is a table dipicting the execution time, for a route from Bloomington,_Indiana to San_Diego,_California, taken on an average of 10 runs.
	---------------------------------------------------------------------------------------------
	|   Routing option 		|        BFS 	   |     DFS 	  |     Uniform 	|		Astar   |
	---------------------------------------------------------------------------------------------
	|      Segments         |      0.0167	   |   0.0275	  | 	 0.844		|		0.992   |
	|      Distance         |      0.0167	   |   0.0275	  | 	 0.804		|		0.167   |
	|      Time             |      0.0167	   |   0.0275	  | 	 0.853		|		0.238   |
	---------------------------------------------------------------------------------------------
	We can notice that routing algorithms for BFS and DFS are same for all routing options (cost function) since, bfs and dfs doesnt return
	an optimal distance and time function.

(3) Which algorithm requires the least memory, and by how much, according to your experiments?
    Least memory is used by DFS since only its successors in a route i.e at a depth is stored, considering b as branching factor in the graph,
    m as the maximum length of the path and d as the depth of the tree. DFS stores at max (b*m) nodes(linear). As for above example,
    average branching factor is 3 and the depth of destination node is 61. hence at max nodes stored will be 183.
    BFS consumes lot of memory since at level b nodes are expanded and further each will expand b nodes and so on, resulting in an exponential
	increase in memory(fringe). For above example, in worst case 3^21 nodes will be stored.
	Uniform and Astar also follows the same approach of BFS, and in worst case both consumes an exponential memory, for example in case of 
	segments both Uniform, astar behaves similar to BFS.

(4) Which heuristic function(s) did you use, how good is it, and how might you make it/them better?
	The problem is a representation of points in a circular plane. Hence heuristic function used here is a straight line distance between each node 
	and goal node. Also, the straight distance between nodes will be a consistent heuristic function so, searching will be faster.
	To calculate straight line distance between two points (coordinate points), haversine forumla is used.
    There are certain cities or points in the graph whose coordinates are not present as part of data to the problem. In such situations,
    haversine distance of its nearest neighbour is used as part of herusitic parameter.
	This heuristic function is used differently for each routing option.
	1. Segments.
	   Smallest distance between all nodes in the graph is used as minimum segment value which is divided by haversine distance is the heuristic
	   function used. We can say Total no of segments is directly proportional to total distance in the map. Therefore, 
	   minimum segment depends on the distance to reach goal node. Hence the heursitic function.
	2. Distance.
	   For distance haversine distance from neighbour node to goal node is used as the heuristic function as mentioned by above reasons.
	   In case of node without coordinates, its nearest neighbour haversine distance is used as a herusitic function. In certain cases, such as 
	   goal node, node without coordinates and its nearest neihgbour are in same line and same order as mentioned, heuristic function does over
	   estimate and returns sub-optimal path. 
	3. Time.
	   Similar to distance, the edge weights here can be considered as time(haversinedistance/max speed of all highways)
	   Max speed is considered since time is inversely proportional to speed and hence max speed will result in least time as a result heuristic 
	   function wouldnt over-estimate.
	4. Longtour.
	   Longtour is implemented similar to distance cost function but checking for a maximum value every time in fringe.
	   Max heap is implemented for longtour detection.
	The costliest heursitic work around used here is for the cities without coordinates. Since, its nearest neighbours are selected as the heruistic
	parameter, there are chances of it overestimating the path. This can be solved by using the actual real values from querying neatgeo coordinate
	server.

Assumptions:
	1. For city data without speed a constant 40 miles per hour is assumed.
	2. Intersection or cities wihtout coordinates, 0.0 is used for latitude and longitude.
	3. If distance and speed is mentioned as 0, a constant 40 miles and 40 miles per hour is added respecitively.

Longtour for uniform and astar is implemented based on greedy approach similar to dijkstra algorithm. For fringe, contrary to minheap implementation
maxheap is used. Optimal solution is not possible using uniform search, since with addition of every node into the heap cost does always increase,
thereby missing out many nodes for future expansion. Hence astar will give a close to optimal solution as we will see ahead using haversine distance
to destination.
'''
import sys
from heapq import heappush, heapify, heappop, _heapify_max
from math import sin, cos, sqrt, atan2, radians

def successors(city):
	return graph[city]['neighbour'].keys()

# Search a path from start_city to end_city, search_type is passed as 0 for BFS and -1 for DFS.
def solve(search_type, start_city, end_city):
	index = 0
	# Fringe is a data structure used as a Stack for DFS and Queue for BFS implementation
	fringe = [(start_city, 0, 0)]
	while len(fringe) > 0:
		next_city = fringe.pop(search_type)
		index += 1
		if next_city[0] == start_city:
			graph[next_city[0]]['parent'] = 'start'
		for s in successors( next_city[0] ):
			# Parent is used as visited matrix. Already visited city is not handled.
			if graph[s]['parent'] != 'null':
				continue
			# Calulcate distance so far
			distance = next_city[1] + graph[next_city[0]]['neighbour'][s][0]
			# Calulcate time so far
			time = next_city[2] + (graph[next_city[0]]['neighbour'][s][0] / float(graph[next_city[0]]['neighbour'][s][1]))
			graph[s]['parent'] = next_city[0]
			if s == end_city:
				return distance, time
			fringe.append((s, distance, time))
	return (0, 0)

# Return a path using uniform search algorithm - Dijkstra algorithm.
def minheap(start_city, end_city, cost_function):
	# Fringe holds the value of (total cost - based on the cost function, distance, time, city)
	fringe = []
	# Min heap is used as priority queue. Least value city is handled first.
	heappush(fringe, (0, 0, 0, start_city))
	visited = []
	distance = 0
	time  = 0
	graph[start_city]['parent'] = 'start'
	while len(fringe) > 0:
		heapify(fringe)
		if cost_function == 'longtour':
			_heapify_max(fringe)
		next_city = heappop(fringe)
		if next_city[3] == end_city:
			return next_city[1], next_city[2]
		visited.append(next_city[3])

		for s in successors( next_city[3] ):
			if s in visited:
				continue
			continue_successor = False
			if cost_function == 'distance':
				neighbour_cost = graph[next_city[3]]['neighbour'][s][0] + next_city[1]
				distance = neighbour_cost
				time = graph[next_city[3]]['neighbour'][s][0] / float(graph[next_city[3]]['neighbour'][s][1]) + next_city[2]
			elif cost_function == 'time':
				neighbour_cost = (graph[next_city[3]]['neighbour'][s][0] / float(graph[next_city[3]]['neighbour'][s][1])) + next_city[2]
				distance = graph[next_city[3]]['neighbour'][s][0] + next_city[1]
				time = neighbour_cost
			elif cost_function == 'segments':
				neighbour_cost = 1 + next_city[0]
				distance = graph[next_city[3]]['neighbour'][s][0] + next_city[1]
				time = graph[next_city[3]]['neighbour'][s][0] / float(graph[next_city[3]]['neighbour'][s][1]) + next_city[2]
			elif cost_function == 'longtour':
				neighbour_cost = next_city[0] + graph[next_city[3]]['neighbour'][s][0] # + (1 / next_city[1])
				distance = neighbour_cost
				time = graph[next_city[3]]['neighbour'][s][0] / float(graph[next_city[3]]['neighbour'][s][1]) + next_city[2]

			continue_successor = False
			for i , (j, k, l, m) in zip(range(len(fringe)), fringe):
				if m == s:
					if cost_function == 'longtour':
						if j > neighbour_cost:
							continue_successor = True
							break
						else:
							del fringe[i]
					else:
						if j < neighbour_cost:
							continue_successor = True
							break
						else:
							del fringe[i]
			if continue_successor:
				continue
			graph[s]['parent'] = next_city[3]
			heappush(fringe, (neighbour_cost, distance, time, s))
	return (0, 0)

# Haversine distance calculates and returns distance from start_city and goal_city coordinates
def haversine(start_city, goal_city):
	city = (0, start_city)
	if graph[start_city]['lat'] == 0.0:
		dist = []
		for n in graph[start_city]['neighbour']:
			if not graph[n]['parent'] == 'temp_visiting':
				dist.append((graph[start_city]['neighbour'][n][0], n))
		heapify(dist)
		if len(dist) == 0:
			return 99999.0
		city = heappop(dist)
		for i in range(0, len(dist)):
			if graph[city[1]]['lat']:
				break
			else:
				heapify(dist)
				city = heappop(dist)
		start_city = city[1]
		if graph[start_city]['lat'] == 0.0:
			orig_parent = graph[start_city]['parent']
			graph[start_city]['parent'] = 'temp_visiting'
			temp_distance = haversine(start_city, goal_city)
			graph[start_city]['parent'] = orig_parent
			return temp_distance

	R = 6373.0
	start_city_latitude = radians(graph[start_city]['lat'])
	start_city_longitude = radians(graph[start_city]['long'])
	goal_city_latitude = radians(graph[goal_city]['lat'])
	goal_city_longitude = radians(graph[goal_city]['long'])
	delta_longitude = goal_city_longitude - start_city_longitude
	delta_latitude = goal_city_latitude - start_city_latitude
	a = sin(delta_latitude / 2)**2 + cos(start_city_latitude) * cos(goal_city_latitude) * sin(delta_longitude / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))
	return (R * c * 0.621371)

# Astar algorithm uses the heruristic function mentioned above for optimal path finding
def astar(start_city, end_city, cost_function):
	fringe = []
	# Min heap is used as priority queue. Least value city is handled first.
	heappush(fringe, (0, 0, 0, start_city, 0))
	graph[start_city]['parent'] = 'start'
	visited = []
	while len(fringe) > 0:
		heapify(fringe)
		if cost_function == 'longtour':
			_heapify_max(fringe)
		next_city = heappop(fringe)

		visited.append(next_city[3])
		if next_city[3] == end_city:
			return next_city[1], next_city[2]

		for s in successors( next_city[3] ):
			if s in visited:
				continue
			continue_successor = False

			heuristic_cost = haversine(s, end_city)

			if cost_function == 'distance':
				neighbour_cost = next_city[1] + graph[next_city[3]]['neighbour'][s][0]
				distance = neighbour_cost
				time = graph[next_city[3]]['neighbour'][s][0] / float(graph[next_city[3]]['neighbour'][s][1]) + next_city[2]

			elif cost_function == 'segments':
				neighbour_cost = next_city[4] + 1
				if heuristic_cost:
					heuristic_cost = min_distance / heuristic_cost
				distance = graph[next_city[3]]['neighbour'][s][0] + next_city[1]
				time = graph[next_city[3]]['neighbour'][s][0] / float(graph[next_city[3]]['neighbour'][s][1]) + next_city[2]

			elif cost_function == 'time':
				neighbour_cost = (graph[next_city[3]]['neighbour'][s][0] / float(graph[next_city[3]]['neighbour'][s][1])) + next_city[2]
				heuristic_cost /= max_speed
				distance = graph[next_city[3]]['neighbour'][s][0] + next_city[1]
				time = neighbour_cost

			elif cost_function == 'longtour':
				neighbour_cost = next_city[1] + graph[next_city[3]]['neighbour'][s][0]
				distance = neighbour_cost
				time = graph[next_city[3]]['neighbour'][s][0] / float(graph[next_city[3]]['neighbour'][s][1]) + next_city[2]

			for i , (j, k, l, m, n) in zip(range(len(fringe)), fringe):
				if m == s:
					if cost_function == 'longtour':
						if j > neighbour_cost + heuristic_cost:
							continue_successor = True
							break
						else:
							del fringe[i]
					else:
						if j < neighbour_cost + heuristic_cost:
							continue_successor = True
							break
						else:
							del fringe[i]
			if continue_successor == True:
				continue
			graph[s]['parent'] = next_city[3]
			heappush(fringe, (neighbour_cost + heuristic_cost, distance, time, s, neighbour_cost))
	return (0, 0)

# Graph data structure is used to store city and segments in graph.
def populate_graph():
	global min_distance
	global max_speed
	city_gps = open("city-gps.txt", "r")
	road_segments = open("road-segments.txt", "r")
	city_data = city_gps.readline().split()
	min_distance = 9999
	# Populate city data
	while city_data:
		graph[city_data[0]] = {}
		graph[city_data[0]].update({'lat' : float(city_data[1])})
		graph[city_data[0]].update({'long' : float(city_data[2])})
		graph[city_data[0]].update({'neighbour': {}})
		city_data = city_gps.readline().split()

	# Populate route data
	route_data = road_segments.readline().split()
	while route_data:
		for city in range(0,2):
			if not route_data[city] in graph:
				graph[route_data[city]] = {'lat' : 0.0, 'long' : 0.0, 'neighbour': {}}

		if len(route_data) == 4:
			route_data.insert(3, 40)
		if int(route_data[2]) == 0:
			route_data[2] = 40
		if int(route_data[3]) == 0:
			route_data[3] = 40

		neighbour_data = {route_data[1]: [int(route_data[2]), int(route_data[3]), route_data[4]]}
		graph[route_data[0]]['neighbour'].update(neighbour_data)

		neighbour_data = {route_data[0]: [int(route_data[2]), int(route_data[3]), route_data[4]]}
		graph[route_data[1]]['neighbour'].update(neighbour_data)

		if int(route_data[3]) > max_speed:
			max_speed = int(route_data[3])
		if min_distance > int(route_data[2]):
			min_distance = int(route_data[2])

		route_data = road_segments.readline().split()

	for node in graph:
		if not 'parent' in node:
			graph[node].update({'parent' : 'null'})

graph = {}
min_distance = 0
max_speed = 0

# Print human and machine readable input
def printSolution(start_city, end_city, distance, time):
	path = []
	parent_city = graph[end_city]['parent']
	path.insert(0, end_city)
	while parent_city != 'start':
		path.insert(0, parent_city)
		parent_city = graph[parent_city]['parent']

	# Human Readable format
	# Formatted according to 15.6 inch screen. Hence, viewed best in 15.6 inch or more screen.
	print("-"*150)
	print("%sStart City%s%sEnd City%s%sDistance%s%sSpeed%s\tTime\tHighway")\
						%(' '*4, ' '*(43 - 4 - len('Start City')), ' '*4, ' '*(43 - 4 - len('End City')), ' '*2, ' '*2, ' '*4, ' '*4)
	print("%s(Miles)%s(Miles/hr)%s(Min)") %(' '*88, ' '*4, ' '*3)
	print("-"*150)
	city = path[0]
	for city_no in range(1, len(path)):
		neighbour_city = path[city_no]
		dist = str(graph[city]['neighbour'][neighbour_city][0])
		speed = graph[city]['neighbour'][neighbour_city][1]
		highway = graph[city]['neighbour'][neighbour_city][2]
		print("%s%s %s%s%s%s%s%s%d%s\t%2.0f\t%s") \
					%(city, ' '*(43 - len(city)),
					  neighbour_city, ' '*(43 - len(neighbour_city)),
					  ' '*5, dist, ' '*(7 - len(dist)),
					  ' '*4, speed, ' '*5,
					  int(dist)/float(speed)*60, highway)
		city = neighbour_city
	print("-"*150)
	print("\n")

	# Machine Readable format
	print distance, time,
	for node in path:
		print(node),
	print "\n"

def verifyInput(start_city, end_city, routing_algorithm, cost_function):
	if start_city not in graph.keys():
		print "Please enter valid start city"
		sys.exit(0)

	if end_city not in graph:
		print "Please enter valid end city"
		sys.exit(0)

	if routing_algorithm not in ("bfs", "uniform", "dfs", "astar"):
		print "Please enter valid routing algorithm"
		sys.exit(0)

	if cost_function not in ("segments", "distance", "time", "longtour"):
		print "Please enter valid cost function"
		sys.exit(0)

	if start_city == end_city:
		print "Please enter different start and end city"
		sys.exit(0)

# Main function
def main():
	start_city = sys.argv[1]
	end_city = sys.argv[2]
	routing_algorithm = sys.argv[3]
	cost_function = sys.argv[4]

	populate_graph()
	verifyInput(start_city, end_city, routing_algorithm, cost_function)

	if routing_algorithm == 'bfs':
		distance, time = solve(0, start_city, end_city)
		if distance and time:
			printSolution(start_city, end_city, distance, time)
			for node in graph:
				graph[node]['parent'] = 'null'
		else:
			print 'didnt find the solution'

	elif routing_algorithm == 'uniform':
		distance, time = minheap(start_city, end_city, cost_function)
		if distance and time:
			printSolution(start_city, end_city, distance, time)
		else:
			print 'didnt find the solution'

	elif routing_algorithm == 'dfs':
		distance, time = solve(-1, start_city, end_city)
		if distance and time:
			printSolution(start_city, end_city, distance, time)
			for node in graph:
				graph[node]['parent'] = 'null'
		else:
			print 'didnt find the solution'

	elif routing_algorithm == 'astar':
		distance, time = astar(start_city, end_city, cost_function)
		if distance and time:
			printSolution(start_city, end_city, distance, time)
		else:
			print 'didnt find the solution'

if __name__ == '__main__':
	main()