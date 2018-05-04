'''
Created on Dec 30, 2015

@author: Alex
'''
import collections
import pandas as pd


class ClusterCandidate:
    def __init__(self):
        self.cluster = None
        self.__is_noise = False
        
    def is_classified(self):
        return self.__is_noise or self.cluster != None
    
    def is_noise(self):
        return self.__is_noise
    
    def set_as_noise(self):
        self.__is_noise = True
    
    def assign_to_cluster(self, cluster):
        self.cluster = cluster
        self.__is_noise = False
        
    def distance_to_candidate(self, other_candidate):
        raise NotImplementedError()
    
class ClusterCandidateIndex:
    def __init__(self, candidates, epsilon):
        self.candidates = candidates
        self.epsilon = epsilon
        
    def find_neighbors_of(self, cluster_candidate):
        neighbors = []
        id = 0
        for item in self.candidates:
            dis = cluster_candidate.distance_to_candidate(item)
            id = id + 1
            if item != cluster_candidate and dis<= self.epsilon and dis!=0:
                neighbors.append(item)          
        return neighbors
    
class Cluster:
    def __init__(self):
        self.members = []
        self.member_set = set()
        
    def add_member(self, item):
        if item in self.member_set:
            raise Exception("item: " + str(item) + " already exists in this cluster")
        self.member_set.add(item)
        self.members.append(item)
        
    def __repr__(self):
        return str(self.members)

    def velocity_statistics(self):
        def get_velocity(item):
            start = item.line_segment.start
            end = item.line_segment.end
            return [start.v, end.v]
        res = list(map(get_velocity, self.members))
        list_v = [item for sublist in res for item in sublist]
        df_v = pd.DataFrame(list_v)
        return df_v

    def angle_histogram(self):
        """
        计算簇中所有点的航向列表
        :return: list所有点的航向
        """
        def get_angles(item):
            start = item.line_segment.start
            end = item.line_segment.end
            return [start.c, end.c]
        res = list(map(get_angles, self.members))
        r = [item for sublist in res for item in sublist]
        return r
        
class ClusterFactory():
    def new_cluster(self):
        return Cluster()
        
def dbscan(cluster_candidates_index, min_neighbors, cluster_factory, getnoise=False):
    clusters = []
    noises = []
    item_queue = collections.deque()

    icount = 0
    for item in cluster_candidates_index.candidates:
        if not item.is_classified():
            print("start neighbors" + str(icount))
            icount = icount + 1
            neighbors = cluster_candidates_index.find_neighbors_of(item)  #找所有近邻
            print("have " + str(len(neighbors)) +  " neighbors")
            if len(neighbors) >= min_neighbors:  #如果近邻个数大于min_neighbors,形成一个簇
                cur_cluster = cluster_factory.new_cluster()
                cur_cluster.add_member(item)
                item.assign_to_cluster(cur_cluster)
                
                for other_item in neighbors:  #把所有找到的近邻都分配到该簇
                    other_item.assign_to_cluster(cur_cluster)
                    cur_cluster.add_member(other_item)
                    item_queue.append(other_item)

                #扩充这个簇
                print("start expand_cluster cluster")
                expand_cluster(item_queue, cur_cluster, min_neighbors, \
                               cluster_candidates_index)
                clusters.append(cur_cluster)
                print("expand cluster have " + str(len(clusters)) + " clusters")
            else:
                item.set_as_noise()

    print("dbscan done")
    if getnoise == True:
        for item in cluster_candidates_index.candidates:
            if item.is_noise() == True:
                noises.append(item)
        return clusters, noises
    else:
        return clusters, None

def expand_cluster(item_queue, cluster, min_neighbors, cluster_candidates_index):
    while len(item_queue) > 0:
        item = item_queue.popleft()
        neighbors = cluster_candidates_index.find_neighbors_of(item)
        if len(neighbors) >= min_neighbors:
            for other_item in neighbors:
                if not other_item.is_classified():
                    item_queue.append(other_item)
                if other_item.is_noise() or not other_item.is_classified():
                    other_item.assign_to_cluster(cluster)
                    cluster.add_member(other_item)
                  
    
    
    
    