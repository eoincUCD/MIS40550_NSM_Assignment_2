"""
************************************************************************************************************************
Project Title: Modelling a Community with Network Software and parkrun Data
************************************************************************************************************************
UCD Assignment Details:
Date Started: 07/03/17
Revision 0
Date Submitted: 23/03/17
Student Name (Student Number): Eoin Carroll (16202781)
Module Code: UCD MIS40550
Module Title: Network Software Modelling
Assessment Title: Network Software Modelling Assignment 2
Module Co-ordinator: Dr James McDermott
************************************************************************************************************************
Instructions:
1.  Copy this python file into a folder
2.  Add a data folder and place data files there
    Link: https://drive.google.com/drive/folders/0B9kelMwrpRsROF9UZFY5SzZZUzg
    Data processing results will be saved in the data folder
3.  Add a results folder
    Run whichever functions you select - make sure you check the run times
    Results will be saved in the results folder
************************************************************************************************************************
Functions included:
generate_weekly_graphs - Generate all weekly graphs, saved separately
combine_networks - Combine a sample into one network. Combine all into full network
erdos_renyi - Create random graphs
graph_properties - Save graph properites to text file in results folder
plot_graphs - Save degree plots to results folder
simulate - Simulate information flow. Save results to data folder
plot_sim - Save simulation plots to results folder
************************************************************************************************************************
Github:
The project was made available on Github with the following URL. Note that the data files are too big to store on Github
https://github.com/eoincUCD/Modelling_Parkrun_Community
************************************************************************************************************************
References:
Week 9 lab solutions
************************************************************************************************************************
"""


import networkx as nx
import pandas as pd
import numpy as np
import time
import datetime
import multiprocessing
import scipy
import random
import matplotlib.pyplot as plt


# Declare Variables
# data_folder = "data/"  This is now hard coded - do not change
# results_folder = "results/"  This is now hard coded - do not change
total_races = 204  # Total number of races available
sample_races = 10  # number of races for small sample, default 10. Ensure this is less than total!!


# Generate all weekly graphs, saved separately
def generate_weekly_graphs(total_races=204):
    # Function to run each weekly graph generation in a separate process - CPU intensive
    # Not perfect but does the job
    print("Generating networks - this will take sometime.")
    startTime = time.time()
    out_file_name = "results/generate_results.txt"
    out_file = open(out_file_name, "w")
    out_file.write("Date and time: " + str(datetime.datetime.now()) + "\n")
    out_file.close()
    jobs = []
    for i in range(total_races):  # Separate process for each dataset
        in_file = "data/" + str(i+1) + ".xlsx"
        p = multiprocessing.Process(target=worker, args=(in_file,out_file_name))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()
    out_file = open(out_file_name, "a")
    out_file.write("Time taken to generate: " + str(time.time() - startTime))  # Log time taken
    out_file.close()


# Used to generate all weekly graphs
def worker(in_file, out_file_name):  # Worker function for multiprocessing generating graphs from each week
    generate(in_file, out_file_name)
    return


# Used to generate all weekly graphs
def generate(in_file, out_file_name):  # Generate a graph from weekly results
    startTime = time.time()

    # Initialise parameters. Note that name and club can be added up to three times, if finished close together
    same_race = 0.005                               # Variable to control weight given to people who ran in same race
    same_name = 0.10                                # Variable to control weight given to people with same second name
    same_club = 0.50                                # Variable to control weight given to people with same club
    close_pos = np.arange(0.000,0.11,0.01)          # Array - closeness weight for finishing close together position
    close_time_limit = 10                           # Set time limit of who to increase weighting (< x seconds)
    close_time_increment = 0.025                    # Array - closeness weight for close together time
    close_time = np.arange(0.000,close_time_limit*close_time_increment,close_time_increment)[::-1]

    G = nx.Graph()

    # Read excel file into pandas dataframe. Data file in format with column names:
    # Pos  ↓, parkrunner  ↓, Time  ↓, Age Cat  ↓, Age Grade  ↓, ↓, Gender Pos  ↓, # Club  ↓, Note  ↓, Total Runs  ↓
    df = pd.read_excel(in_file)

    # Read number of rows in dataframe
    entries = df.shape[0]
    # entries = 10 todo - make sure this option is off

    for i in range(entries):                 # Iterate through rows
        j = i-1                              # Connecting previous nodes working backwards through spreadsheet dataframe
        while(j>=0):
            if df.iloc[j, 1] != "Unknown" and df.iloc[i, 1] != "Unknown":                          # Ignore unknown rows
                if (df.iloc[i,1] != df.iloc[j,1]):                          # If name duplicate, skip adding new weight
                    G.add_edge(df.iloc[i,1], df.iloc[j,1], weight=same_race) # Add basic connection for same race

                    if df.iloc[i,1].split()[1] ==  df.iloc[j,1].split()[1]:  # Same second name
                        new_weight = G[df.iloc[i,1]][df.iloc[j,1]]['weight'] + same_name  # Include same name parameter
                        G.add_edge(df.iloc[i, 1], df.iloc[j, 1], weight=new_weight)

                    if df.iloc[i,7] != "" and df.iloc[i,7] ==  df.iloc[j,7]:  # Same club and club not blank
                        new_weight = G[df.iloc[i,1]][df.iloc[j,1]]['weight'] + same_club  # Include same club parameter
                        G.add_edge(df.iloc[i, 1], df.iloc[j, 1], weight=new_weight)

                    if df.iloc[i,0] - df.iloc[j,0] < 10:                      # Finish position close together
                        new_weight = G[df.iloc[i, 1]][df.iloc[j, 1]]['weight'] + close_pos[df.iloc[i,0] - df.iloc[j,0]]
                        G.add_edge(df.iloc[i, 1], df.iloc[j, 1], weight=new_weight)

                        if df.iloc[i, 1].split()[1] == df.iloc[j, 1].split()[1]:  # Same second name
                            new_weight = G[df.iloc[i, 1]][df.iloc[j, 1]]['weight'] + same_name  # Include same name
                            G.add_edge(df.iloc[i, 1], df.iloc[j, 1], weight=new_weight)

                        if df.iloc[i, 7] != "" and df.iloc[i, 7] == df.iloc[j, 7]:  # Same club and club not blank
                            new_weight = G[df.iloc[i, 1]][df.iloc[j, 1]]['weight'] + same_club  # Include same club
                            G.add_edge(df.iloc[i, 1], df.iloc[j, 1], weight=new_weight)

                    if abs(get_secs(df.iloc[i, 2])) - abs(get_secs(df.iloc[j, 2])) < close_time_limit:
                        # Finish time close together
                        new_weight = G[df.iloc[i, 1]][df.iloc[j, 1]]['weight'] + \
                                     close_time[get_secs(df.iloc[i, 2]) - get_secs(df.iloc[j, 2])]
                        G.add_edge(df.iloc[i, 1], df.iloc[j, 1], weight=new_weight)
                        if df.iloc[i, 1].split()[1] == df.iloc[j, 1].split()[1]:  # Same second name
                            new_weight = G[df.iloc[i, 1]][df.iloc[j, 1]]['weight'] + same_name  # Include same name
                            G.add_edge(df.iloc[i, 1], df.iloc[j, 1], weight=new_weight)

                        if df.iloc[i, 7] != "" and df.iloc[i, 7] == df.iloc[j, 7]:  # Same club and club not blank
                            new_weight = G[df.iloc[i, 1]][df.iloc[j, 1]]['weight'] + same_club  # Include same club
                            G.add_edge(df.iloc[i, 1], df.iloc[j, 1], weight=new_weight)

            j -= 1

    endTime = time.time()
    edgelist_name = in_file.split('.')[0] + "_network" + ".csv"  # Only use number of file and csv
    nx.write_edgelist(G, edgelist_name, delimiter=',', data=['weight'])
    file_print = in_file.split('.')[0].split('/')[1] + ": {0:0.2f} seconds.".format(endTime - startTime)
    file_print = file_print + str(G.number_of_nodes()) + " nodes." + str(G.number_of_edges()) + " edges."
    out_file = open(out_file_name, "a")
    out_file.write(file_print + "\n")
    out_file.close()
    return


# Used to generate all weekly graphs
def get_secs(t):  # converts the time entry into seconds
    # This function converts the time entry into seconds.
    # Note there are 3 time formats handled - "dd/mm/yyyy hh:mm:ss", "hh:mm:ss" and "mm:ss:00"
    # Assume that we will never exceed 24hrs.
    split_main = str(t).split(' ')
    if len(split_main) == 2:                       # Time format "dd/mm/yyyy hh:mm:ss"
        split_day = split_main[0].split('-')
        split_sub = split_main[1].split(':')
        h = 0
        m = int(split_sub[0]) + int(split_day[2])*24
        s = int(split_sub[1])
    else:                                          # Time format "hh:mm:ss" or "mm:ss:00"
        split_sub = split_main[0].split(':')
        if int(split_sub[0]) > 1:                  # Time format is mm:ss:00. Assume h can't be > 1
            h = 0
            m = int(split_sub[0])
            s = int(split_sub[1])
        else:                                      # Time format is hh:mm:ss
            h = int(split_sub[0])
            m = int(split_sub[1])
            s = int(split_sub[2])
    # print("h,m,s", h,m,s)
    return h * 3600 + m * 60 + s


# Combine all weekly graphs
def combine_networks(total_races=204, sample_races=10):  # Combine graphs and save as CSV network
    G = nx.Graph()
    print("Combining sample network.")
    out_file_name = "results/combine_sample.txt"
    out_file = open(out_file_name, "w")
    out_file.write("Date and time: " + str(datetime.datetime.now()) + "\n")
    startTime = time.time()
    for i in range(sample_races):
        infile = "data/" + str(i+1) + "_network.csv"
        G = add_network(G, infile)
        out_file.write("Finished " + str(i+1) + "\n")

    nx.write_edgelist(G, "data/" + "sample_network.csv", delimiter=',', data=['weight'])
    out_file.write("Time taken to combine: " + str(time.time() - startTime))
    out_file.close()

    G = nx.Graph()
    print("Combining full network.")
    out_file_name = "results/combine_full.txt"
    out_file = open(out_file_name, "w")
    out_file.write("Date and time: " + str(datetime.datetime.now()) + "\n")    
    startTime = time.time()
    for i in range(total_races):
        infile = "data/" + str(i+1) + "_network.csv"
        G = add_network(G, infile)
        out_file.write("Finished " + str(i+1) + "\n")

    nx.write_edgelist(G, "data/" + "full_network.csv", delimiter=',', data=['weight'])
    out_file.write("Time taken to combine: " + str(time.time() - startTime))
    out_file.close()


# Used to combine all weekly graphs
def add_network(G, infile):  # Add a network to graph G in RAM
    K = nx.read_edgelist(infile, delimiter=",", data=(('weight', float),))
    for edge in nx.edges(K):
        if G.has_edge(edge[0],edge[1]):
            new_weight = (G[edge[0]][edge[1]]['weight']) + (K[edge[0]][edge[1]]['weight'])
            G.add_edge(edge[0],edge[1], weight=new_weight)
        else:
            G.add_edge(edge[0], edge[1], weight=K[edge[0]][edge[1]]['weight'])
    return G


# Convert weights to a probability less than 1 of two people knowing each other
def scale(x):  # scale to never reach 1 (100% probability)
    y = 1 - 1/(1+x)
    return y


# Create erdos_renyi graphs of multiple sizes
def erdos_renyi():
    random.seed(12345)
    x = 10
    N = (100, 1000, 5000)  # 5000 was also included to create large graph
    for n in N:
        pn = n / 10000
        G = nx.erdos_renyi_graph(n, pn)

        for edge in G.edges():
            G.add_edge(edge[0], edge[1], weight=random.expovariate(x))
            #print(G[edge[0]][edge[1]]['weight'])

        # Save to disk
        out_file = "data/erdos_" + str(n) + "_network.csv"
        nx.write_edgelist(G, out_file, delimiter=',', data=['weight'])
        print("Erdos_renyi graph created as", out_file)
    return


# Save graph properties of input graph
def graph_properties(in_file_name):
    startTime = time.time()
    G = nx.read_edgelist("data/" + in_file_name, delimiter=",", data=(('weight', float),))  # Read graph

    out_file_name = "results/" + "properties_" + in_file_name + ".txt"
    out_file = open(out_file_name, "w")

    out_file.write("Date and time: " + str(datetime.datetime.now()) + "\n")
    out_file.write(in_file_name + " properties: " + "\n")
    out_file.write("Order: " + str(G.order()) + "\n")
    out_file.write("Size: " + str(G.size()) + "\n")
    # out_file.write("Degree:" + G.degree() + "\n")  # Too large to out_file.write out
    out_file.write("Density: " + str(G.size() / scipy.misc.comb(G.order(), 2)) + "\n")
    out_file.write("Clustering coefficient: " + str(nx.average_clustering(G)) + "\n")
    try:
        out_file.write("Diameter: " + str(nx.diameter(G)) + "\n")
    except:
        out_file.write("Diameter: unknown" + "\n")
    out_file.write("Size of largest component: " + str(max(len(c) for c in nx.connected_components(G))) + "\n")
    out_file.write("Time taken to complete graph properties function: " + str(time.time()-startTime))

    out_file.close()

    print("Properties saved for", out_file_name)
    return


# Save plots of input graph
def plot_graphs(in_file):
    in_file_name = "data/" + in_file
    out_file = "results/" + "degree_histogram_" + in_file + ".png"
    G = nx.read_edgelist(in_file_name, delimiter=",", data=(('weight', float),))
    plot_title = "Degree Histogram (All connections) for " + in_file
    plot_degree(G, out_file, plot_title)  # Plot unadjusted degree graph
    for edge in G.edges():  # Remove any edges that are less than 0.5 probability of knowing each other
        if scale(G[edge[0]][edge[1]]["weight"]) < 0.5:
            G.remove_edge(edge[0], edge[1])
    out_file = "results/" + "degree_histogram_adjusted_" + in_file + ".png"
    plot_title = "Degree Histogram (P > 0.5) for " + in_file
    plot_degree(G, out_file, plot_title)  # Plot adjusted graph only for greater than 0.5 probability
    print("Plots saved for " + in_file)


# Used in plotting
def plot_degree(G, out_file, plot_title):
    degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
    # Plot histogram
    plt.figure()
    plt.hist(degree_sequence, bins=30, color='#3399ff')
    plt.title(plot_title)
    plt.ylabel("Number of Nodes (Network total: " + str(G.order()) + ")")
    plt.xlabel("Degree")
    plt.savefig(out_file)
    # plt.show()


# Save simulation of input graph
def simulate(in_file, maxits):
    random.seed(12345)
    in_file_name = "data/" + in_file
    out_file_name = "data/" + "simulate_" + in_file

    G = nx.read_edgelist(in_file_name, delimiter=",", data=(('weight', float),))

    largest_node = "NULL"
    largest_node_size = 0
    for n in G.nodes():   # Find node with most connections - message starts here
        G.node[n]["state"] = 0
        if len(G.neighbors(n)) > largest_node_size:
            largest_node = n
            largest_node_size = len(G.neighbors(n))

    G.node[largest_node]["state"] = 1  # Set node of most connections state to 1

    out_file = open(out_file_name, "w")  # Store initial states
    for n in G.nodes():
        out_file.write(str(G.node[n]["state"]) + ",")
    out_file.close()

    old_state = {}  # used to stop iteration loop if there is no state change
    x = 0
    for i in range(maxits):  # Loop through each step for x number of iterations max
        G = step(G, out_file_name)  # Complete step
        new_state = {}
        for u in G.nodes():
            new_state[u] = G.node[u]["state"]
        if old_state == new_state:  # Check for steady state
            x = i
            break
        old_state = new_state

    number_informed = 0
    for u in G.nodes():
        if G.node[u]["state"] == 2:
            number_informed += 1

    print("Simulation saved for", out_file_name)
    print("Iterations to steady state:", x, ". . . Number informed: ", number_informed, "/", str(G.order()))
    return


# Used in simulation
def step(G, out_file_name):
    new_state = {}
    random.seed(12345)
    for u, d in G.nodes(data=True):
        if G.node[u]["state"] == 1:  # If a node has received the message, do something
            for v in G.neighbors(u):
                if G.node[v]["state"] == 0:  # If that nodes neighbour is state 0, maybe pass on message
                    if scale(G[v][u]["weight"]) > random.random():  # Pass on message based on weight probability
                        # todo erdis remi graphs should not be using scale
                        new_state[v] = 1
            new_state[u] = 2  # That node is now finished
        else:
            new_state[u] = G.node[u]["state"]  # Used if node was not changed

    for u in G.nodes():  # Update G with new states
        G.node[u]["state"] = new_state[u]

    out_file = open(out_file_name, "a")  # Save states as a row in CSV
    out_file.write("\n")
    for u in G.nodes():
        out_file.write(str(G.node[u]["state"]) + ",")
    out_file.close()

    # save node name and state, csv
    return G


# Plot simulation results
def plot_sim(in_file):
    in_file_name = "data/simulate_" + in_file
    df = pd.read_csv(in_file_name,sep=',',header=None)

    state_0 = [0] * len(df)
    state_1 = [0] * len(df)
    state_2 = [0] * len(df)

    for i in range(len(df)):
        for j in df.iloc[i].values[:]:
            if j == 0:
                state_0[i] += 1
            if j == 1:
                state_1[i] += 1
            if j == 2:
                state_2[i] += 1

    plot_title = "State change over time iteration " + in_file
    out_file = "results/" + "simulation_plot_" + in_file + ".png"
    plt.figure()
    plt.plot(state_0, label='State 0 (No Message)')
    plt.plot(state_1, label='State 1 (Message Received)')
    plt.plot(state_2, label='State 2 (Inactive)')
    plt.legend()
    plt.title(plot_title)
    plt.ylabel("Number of Nodes")
    plt.xlabel("Time Iterations")
    plt.savefig(out_file)
    # plt.show()

    print("Simulation plot saved for", in_file)
    return


if __name__ == "__main__":
    # This section of code was used to generate the full network, advised not to run unless you have 3+ hours
    print("Step 1/8 - Generate parkrun networks from raw data:")
    print("Parkrun networks generated previously.")
    # generate_weekly_graphs(total_races)  # Takes 4 hours
    print("")

    # This section creates a sample of default 10 races and also the full network for all races - takes ~10mins
    print("Step 2/8 - Combine and save parkrun networks:")
    print("Parkrun networks combined previously.")
    # combine_networks(total_races, sample_races)  # Takes 5 mins
    print("")

    print("Step 3/8 - Generate random Erdos Remi graphs and save:")
    print("Erdos Remi networks created previously.")
    # erdos_renyi()  # Takes 20mins or so I think?? Need to confirm
    print("")

    print("Step 4/8 - Read networks and examine properties:")
    # graph_properties("1_network.csv")  # Save properties of first race
    # graph_properties("sample_network.csv")  # Save properties of sample races - takes 5 mins
    # graph_properties("full_network.csv")  # Save properties of full network - takes 10 hours
    # graph_properties("erdos_100_network.csv")  # Save properties of erdos_100
    # graph_properties("erdos_1000_network.csv")  # Save properties of erdos_1000
    # graph_properties("erdos_5000_network.csv")  # Save properties of erdos_5000 - takes 5 hours
    print("")

    print("Step 5/8 - Plot networks:")
    # plot_graphs("1_network.csv")
    # plot_graphs("sample_network.csv")
    # plot_graphs("full_network.csv")  # Takes 5mins
    # plot_graphs("erdos_100_network.csv")
    # plot_graphs("erdos_1000_network.csv")
    # plot_graphs("erdos_5000_network.csv")  # Takes 10mins
    print("")

    print("Step 6/8 - Simulate Information Flow:")
    # simulate("1_network.csv", 20)
    # simulate("sample_network.csv", 20)
    # simulate("full_network.csv", 20)  # Takes 5 mins
    # simulate("erdos_100_network.csv", 20)
    # simulate("erdos_1000_network.csv", 20)
    # simulate("erdos_5000_network.csv", 20)  # Takes 10 mins
    print("")

    print("Step 7/7 - Save Simulation Plots:")
    # plot_sim("1_network.csv")
    # plot_sim("sample_network.csv")
    # plot_sim("full_network.csv")
    # plot_sim("erdos_100_network.csv")
    # plot_sim("erdos_1000_network.csv")
    # plot_sim("erdos_5000_network.csv")