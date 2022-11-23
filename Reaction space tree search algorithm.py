def SHITTS1(initialSMILE, finalSMILE):
    t = time.time()
    currentSMILE = initialSMILE # the current "best" molecule chosen to be the optimal next molecule in the reaction
    products_pathway = [initialSMILE] # list containing all the molecules that are in the reaction
    reactions_pathway = [] # list of all the reactions involved in the synthesis
    actions = AlkeneReactions.getAllReactions()
    

    while evaluate1(currentSMILE, finalSMILE) != 1: # repeat this process while the currentSMILE is not the finalSMILE
        best_action = []
        best_node = [currentSMILE]
        top_node_score = 0 # the score obtained from the child node with the highest score from the simulations

        for i in range(len(actions)): # iterates through all possible child nodes
            child_nodes = actions[i](currentSMILE)  
            random_i1 = random.randrange(len(child_nodes))
            child_node = child_nodes[random_i1] # creates a child node from currentSMILE
            child_node_score = 0 # the score for each node, tallied from results of each simulation
            
            if evaluate1(child_node, finalSMILE) == 1: # if the child node is finalSMILE, break out of loop
                best_action.append(actions[i])
                best_node.append(child_node)
                break
            if evaluate1(child_node, currentSMILE) == 1: 
                # move to next child node if the current one is not different from currentSMILE
                continue
            if (evaluate1(child_node, finalSMILE) != 1) and (evaluate1(child_node, currentSMILE) != 1):
                # if neither of the previous conditions are met, proceed to run simulations, each with max num of rollouts
                simulations_count = 0 # number of simulations run from each child node
                actions1 = AlkeneReactions.getAllReactions() # all reactions possible from a given child node

                while simulations_count < 20:
                    rollouts_count = 0 # number of reactions that each simulation from a child node can run for
                    random_i2 = random.randrange(len(actions1)) # creates random index to make random selection of reaction
                    leaf_nodes1 = actions1[random_i2](child_node)
                    random_i21 = random.randrange(len(leaf_nodes1))
                    leaf_node1 = leaf_nodes1[random_i21] # selects a random leaf node from child node

                    if evaluate1(leaf_node1, finalSMILE) == 1:
                        # if leaf node results in "win", increase child node's score and continue simulations
                        child_node_score += 1000000
                        simulations_count += 1
                        continue
                    if evaluate1(leaf_node1, child_node) == 1:
                        # if the two nodes are the same, remove that particular reaction from actions1,
                        # don't count as a simulation
                        # simulations_count -= 1
                        actions1.remove(actions1[random_i2])
                        # print("actions1 is", len(actions1))
                        if len(actions1) == 0:
                            # actions1 = AlkeneReactions.getAllReactions()
                            break
                        simulations_count += 1
                        continue                        
                      
                    
                    if (evaluate1(leaf_node1, finalSMILE) != 1) and (evaluate1(leaf_node1, child_node) != 1):
                        # if neither of the previous conditions are met, do the following
                        actions2 = AlkeneReactions.getAllReactions()
                        random_i3 = random.randrange(len(actions2))
                        # print("random_i3 is", random_i3)
                        leaf_nodes2 = actions2[random_i3](leaf_node1)
                        random_i31 = random.randrange(len(leaf_nodes2))
                        leaf_node2 = leaf_nodes2[random_i31]

                        while rollouts_count < 20: 
                            # print("actions2 is ", len(actions2))
                            if evaluate1(leaf_node2, finalSMILE) == 1:
                                # if this is true, then increase child node's score and continue simulations
                                child_node_score += 1000
                                break
                            if evaluate1(leaf_node2, leaf_node1) == 1:
                                # if  no change, then select a new leafnode2, and remove that particular reaction
                                # rollouts_count -= 1
                                actions2.remove(actions2[random_i3])
                                if len(actions2) == 0:
                                    break
                                random_i3 = random.randrange(len(actions2))
                                # print("random_i3 is", random_i3)
                                leaf_nodes2 = actions2[random_i3](leaf_node1)
                                random_i31 = random.randrange(len(leaf_nodes2))
                                leaf_node2 = leaf_nodes2[random_i31]
                                
                                continue
                            if (evaluate1(leaf_node2, finalSMILE) != 1) and (evaluate1(leaf_node2, leaf_node1) != 1):
                                # if neither of the two conditions are met, do the following
                                random_i4 = random.randrange(len(actions2))
                                leaf_nodes3 = actions2[random_i4](leaf_node2)
                                random_i41 = random.randrange(len(leaf_nodes3))
                                leaf_node2 = leaf_nodes3[random_i41]
                                actions2 = AlkeneReactions.getAllReactions()
                                rollouts_count += 1

                        simulations_count += 1

                if child_node_score > top_node_score:
                    top_node_score = child_node_score
                    best_node.append(child_node)
                    best_action.append(actions[i])


        if evaluate1(best_node[-1], currentSMILE) != 1:
            products_pathway.append(Chem.MolToSmiles(Chem.MolFromSmiles(best_node[-1])))
        currentSMILE = Chem.MolToSmiles(Chem.MolFromSmiles(best_node[-1]))
        if len(best_action) != 0:
            reactions_pathway.append(best_action[-1])
    print(f'{time.time() - t:.5f} sec')
    print("The reactions pathway is", reactions_pathway)
    print("The products formed are", products_pathway)
