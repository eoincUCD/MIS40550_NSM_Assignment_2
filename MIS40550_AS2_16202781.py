"""
************************************************************************************************************************
Project Title: Modelling a Community with Network Software and Parkrun Data
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
Only run generate by itself - multiprocessing is not perfectly implemented
************************************************************************************************************************
Functions included:
************************************************************************************************************************
Notes:
************************************************************************************************************************
References
Week 9 lab solutions
************************************************************************************************************************
"""


import networkx as nx
import pandas
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
    for i in range(total_races):
        in_file = "data/" + str(i+1) + ".xlsx"
        p = multiprocessing.Process(target=worker, args=(in_file,out_file_name))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()
    out_file = open(out_file_name, "a")
    out_file.write("Time taken to generate: " + str(time.time() - startTime))
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
    close_pos = np.arange(0.000,0.11,0.01)        # Array - closeness weight for finishing close together position
    close_time_limit = 10                           # Set time limit of who to increase weighting (< x seconds)
    close_time_increment = 0.025
    # Array - closeness weight for close together time
    close_time = np.arange(0.000,close_time_limit*close_time_increment,close_time_increment)[::-1]

    G = nx.Graph()

    # Read excel file into pandas dataframe. Data file in format with column names:
    # Pos  ↓, parkrunner  ↓, Time  ↓, Age Cat  ↓, Age Grade  ↓, ↓, Gender Pos  ↓, # Club  ↓, Note  ↓, Total Runs  ↓
    df = pandas.read_excel(in_file)

    # Read number of rows in dataframe
    # entries = df.shape[0]
    entries = 10

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

                    # Scale weight to never reach 1 (100% probability of knowing each other)
                    # Not scaling at this point
                    # scaled_weight = scale(G[df.iloc[i,1]][df.iloc[j,1]]['weight'])
                    # G.add_edge(df.iloc[i, 1], df.iloc[j, 1], weight=scaled_weight)

                    # print(i, df.iloc[i, 1], df.iloc[j, 1], G[df.iloc[i, 1]][df.iloc[j, 1]]['weight'])
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


# Convert weights to a probability less than 1 of two preople knowing each other
def scale(x):  # scale to never reach 1 (100% probability)
    y = 1 - 1/(1+x)
    return y


# Create erdos_renyi graphs of multiple sizes
def erdos_renyi():
    random.seed(12345)
    x = 10
    N = (100, 1000)  # 5000 was also included to create large graph
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


# Print graph properties of input graph
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


def plot_all(in_file):
    in_file_name = "data/" + in_file
    out_file = "results/" + "degree_histogram_" + in_file + ".png"
    G = nx.read_edgelist(in_file_name, delimiter=",", data=(('weight', float),))
    plot_title = "Unadjusted Degree Histogram for " + in_file
    plot_degree(G, out_file, plot_title)
    for edge in G.edges():
        if G[edge[0]][edge[1]]["weight"] < 0.5:
            G.remove_edge(edge[0], edge[1])
    out_file = "results/" + "degree_histogram_adjusted_" + in_file + ".png"
    plot_title = "Adjusted Degree Histogram for " + in_file
    plot_degree(G, out_file, plot_title)
    print("Plots saved for " + in_file)


def plot_degree(G, out_file, plot_title):
    degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
    # Plot histogram
    plt.figure()
    plt.hist(degree_sequence, bins=20, color='blue')
    plt.title(plot_title)
    plt.ylabel("Number of Nodes")
    plt.xlabel("Degree")
    plt.savefig(out_file)
    # plt.show()


if __name__ == "__main__":
    # This section of code was used to generate the full network, advised not to run unless you have 3+ hours
    print("Step 1/8 - Generate parkrun networks from raw data:")
    print("Parkrun networks generated previously.")
    # generate_weekly_graphs(total_races)
    print("")

    # This section creates a sample of default 10 races and also the full network for all races - takes ~10mins
    print("Step 2/8 - Combine and save parkrun networks:")
    print("Parkrun networks combined previously.")
    # combine_networks(total_races, sample_races)
    print("")

    print("Step 3/8 - Generate random Erdos Remi graphs and save:")
    # erdos_renyi()
    print("Erdos Remi networks created previously.")
    print("")

    print("Step 4/8 - Read networks and examine properties:")
    graph_properties("1_network.csv")  # Save properties of first race
    graph_properties("sample_network.csv")  # Save properties of sample races - takes 5 mins
    # graph_properties("full_network.csv")  # Save properties of full network - takes 10 hours
    graph_properties("erdos_100_network.csv")  # Save properties of erdos_100
    graph_properties("erdos_1000_network.csv")  # Save properties of erdos_1000
    # graph_properties("erdos_5000_network.csv")  # Save properties of erdos_10000 - takes 5 hours
    print("")

    print("Step 5/8 - Plot networks:")
    plot_all("1_network.csv")
    plot_all("sample_network.csv")
    plot_all("full_network.csv")  # Takes 5mins
    plot_all("erdos_100_network.csv")
    plot_all("erdos_1000_network.csv")
    plot_all("erdos_5000_network.csv")  # Takes 10mins

    # print("Step 7/8 - Simulate Information Flow:")
    # todo
    # simulate flow and print results



