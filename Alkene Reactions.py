import pandas as pd
import numpy as np
import random
import time
from collections import defaultdict

# Commented out IPython magic to ensure Python compatibility.
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdchem

import re
import matplotlib.pyplot as plt
# %matplotlib inline


molecule = Chem.MolFromSmiles("N[C@@H](CCC(N[C@@H](CS)C(NCC(O)=O)=O)=O)C(O)=O")

fig = Draw.MolToMPL(molecule, size=(200, 200))

def draw(smile):
    molecule = Chem.MolFromSmiles(smile)
    fig = Draw.MolToMPL(molecule, size = (200,200))


from rdkit import DataStructs

ms = [Chem.MolFromSmiles('Clc1ccccc1'), Chem.MolFromSmiles('c1ccccc1Br'), Chem.MolFromSmiles('COC')]

fps = [Chem.RDKFingerprint(x) for x in ms]
DataStructs.FingerprintSimilarity(fps[0],fps[1])

# this can be used to convert Kekule version of Smiles from ChemDraw into the canonical, non-Kekule version of the Smiles string


Chem.MolToSmiles(Chem.MolFromSmiles('[H]C=C([H])C'))

class FunctionalGroups:

    # This class is used to identify functional groups that exist within a given molecule. 

    @staticmethod
    def alkene(smile):

        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts("[C:1]=[C:2]")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def internal_alkyne(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts('CC#CC')
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def terminal_alkyne(smile):
        m = Chem.MolFromSmiles(smile)
        # patt = Chem.MolFromSmarts("[C;$(C#C)][H]")
        patt = Chem.MolFromSmarts("[C]#[CH]")
        FG_found = m.HasSubstructMatch(patt)
        

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def acetylide_anion(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts("[C]#[C-]")
        FG_found = m.HasSubstructMatch(patt)
        

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def benzene(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts("c1ccccc1")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def amine(smile):
        m = Chem.MolFromSmiles(smile)
        patt1 = Chem.MolFromSmarts("NC")
        patt2 = Chem.MolFromSmarts("Nc")
        FG_found1 = m.HasSubstructMatch(patt1)
        FG_found2 = m.HasSubstructMatch(patt2)

        if FG_found1 == True or FG_found2 == True:
            return True
        else:
            return False

    @staticmethod
    def alcohol(smile):
        m = Chem.MolFromSmiles(smile)
        patt1 = Chem.MolFromSmiles("O[H]")
        FG_found1 = m.GetSubstructMatches(patt1)

        patt2 = Chem.MolFromSmiles("[H]OC=O")
        FG_found2 = m.GetSubstructMatches(patt2)

 
        if len(FG_found1) > len(FG_found2):
            return True
        else:
            return False

    @staticmethod
    def ether(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmiles("COC")

    
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def aldehyde(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmiles("O=CC")
    
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def ketone(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmiles("CC(C)=O")
    
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def alkylhalide(smile):
        m = Chem.MolFromSmiles(smile)
        patt1 = Chem.MolFromSmiles("Br")
        patt2 = Chem.MolFromSmiles("Cl")
        patt3 = Chem.MolFromSmiles("F")
        patt4 = Chem.MolFromSmiles("I")

        FG_found1 = m.HasSubstructMatch(patt1)
        FG_found2 = m.HasSubstructMatch(patt2)
        FG_found3 = m.HasSubstructMatch(patt3)
        FG_found4 = m.HasSubstructMatch(patt4)
    
        if FG_found1 == True or FG_found2 == True or FG_found3 == True or FG_found4 == True:
            return True
        else:
            return False

    @staticmethod
    def thiol(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmiles("CS[H]")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def carboxylicacid(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmiles("CC(O)=O")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def ester(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmiles("CC(OC)=O")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def amide(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmiles("C(N)=O")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def epoxide(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmiles("C1CO1")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def ring_alkene(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts("[C;R]=[C;R]")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def nonring_alkene(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts("[C;!R]=[C;!R]")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False
    
    @staticmethod
    def cleaved_alkene(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts("[C]=[100*]")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False
    
    @staticmethod
    def non_ring_1_2_diols(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts("[C;!R]([O])-[C;!R]([O])")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def ring_1_2_diols(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts("[C;R]([O])-[C;R]([O])")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

    @staticmethod
    def cleaved_ring_vicinal_diols(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts("[C]([100*])[O]") 
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False


class AlkeneReactions:

    @staticmethod
    def HBr_addition(smile):
        smile_products = [smile] # this list will contain all possible products, accounting for stereochemistry
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
        
            # We must consider regioselectivity for hydrogen halide addition to an alkene. The halide will add to the more
            # substituted alkene carbon and the hydrogen will add to the less-substituted one. In the case where both alkene
            # carbons have equal number of carbon substituents, two possible products can form since the halide can bond
            # to either carbon.

            # one alkene carbon has two carbons singly-bonded to it; the other alkene carbon is not bonded to any carbons
            rxn1 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:4])=[CH2:2].[Br:3]>>[C:0][C:1]([C:4])(-[Br:3])[CH2:2]', useSmiles=True)        

            # This rxn format takes in a SMARTS code that can find, generally, alkenes of a certain degree of substitution. 
            # ex. In rxn1, [CH0:1]([C:4])=[CH2:2] indicates that a disubstituted alkene carbon, "C1" with 0 H bonded to it, 
            # is doubly-bonded to a nonsubstituted alkene carbon, "C2" with 2 H's bonded to it. The "." separates two
            # unbonded compounds, in this case, the alkene and the bromine. The reaction symbol ">>" then indicates reaction,
            # from which results the halogen addition product.

            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Br]")))
            # This ps format gives the SMILES output of the reaction product between the SMILES input and HBr.

            # one alkene carbon is singly-bonded to two carbons, the other is singly-bonded to one carbon
            rxn2 = AllChem.ReactionFromSmarts('[C:6][C:1]([C:4])=[CH:2]([C:5]).[Br:3]>>[C:6][C:1]([C:4])(-[Br:3])[CH:2]([C:5])', useSmiles=True)
            ps2 = rxn2.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Br]")))

            # both alkene carbons are each singly-bonded to two carbons 
            rxn3 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:2])=[C:3]([C:4])[C:5].[Br:6]>>[C:0][C:1]([C:2])(-[Br:6])[C:3]([C:4])[C:5]', useSmiles=True)
            ps3 = rxn3.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Br]")))

            # one alkene carbon is singly-bonded to one carbon, the other is not bonded to any carbons
            rxn4 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH2:2].[Br:3]>>[C:4][CH:1](-[Br:3])-[CH2:2]', useSmiles=True)
            ps4 = rxn4.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Br]")))

            # both alkene carbons each singly-bonded to one carbon
            rxn5 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH:2][C:5].[Br:3]>>[C:4][CH:1](-[Br:3])[CH:2][C:5]', useSmiles=True)
            ps5 = rxn5.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Br]")))

            # both alkene carbons are not bonded to any carbons
            rxn6 = AllChem.ReactionFromSmarts('[CH2:1]=[CH2:2].[Br:3]>>[CH2:1](-[Br:3])[CH3:2]', useSmiles=True)
            ps6 = rxn6.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Br]")))

            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("[C][C]([C])=[CH2]")
            r1 = m1.HasSubstructMatch(patt1)

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[C][C]([C])=[CH][C]")
            r2 = m2.HasSubstructMatch(patt2)

            m3 = Chem.MolFromSmiles(smile1)
            patt3 = Chem.MolFromSmarts("[C][C]([C])=[C]([C])[C]")
            r3 = m3.HasSubstructMatch(patt3)

            m4 = Chem.MolFromSmiles(smile1)
            patt4 = Chem.MolFromSmarts("[C][CH]=[CH2]")
            r4 = m4.HasSubstructMatch(patt4)

            m5 = Chem.MolFromSmiles(smile1)
            patt5 = Chem.MolFromSmarts("[C][CH]=[CH][C]")
            r5 = m5.HasSubstructMatch(patt5)

            m6 = Chem.MolFromSmiles(smile1)
            patt6 = Chem.MolFromSmarts("[CH2]=[CH2]")
            r6 = m6.HasSubstructMatch(patt6)

            alkene_H_halide_reactions_list = [rxn1, rxn2, rxn3, rxn4, rxn5, rxn6] 
            H_halideaddition_products_list = [ps1, ps2, ps3, ps4, ps5, ps6]
            alkene_regiochemistry_list = [r1, r2, r3, r4, r5, r6]
            num_possible_products_list = [1, 1, 2, 1, 2, 1]
        
            i = 0
            for i in range(6):
                if alkene_regiochemistry_list[i] == True:
                    j = 0
                    for j in range(num_possible_products_list[i]):
                        smile_products.append(Chem.MolToSmiles(H_halideaddition_products_list[i][j][0]))
                        j += 1
                    del(smile_products[0])
                    smile1 = smile_products[0]
                    
                    i = 0
                    break
            
                i += 1

        return smile_products


    @staticmethod
    def HCl_addition(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:

            # one alkene carbon has two carbons singly-bonded to it; the other alkene carbon is not bonded to any carbons
            rxn1 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:4])=[CH2:2].[Cl:3]>>[C:0][C:1]([C:4])(-[Cl:3])[CH2:2]', useSmiles=True)        
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Cl]")))

            # one alkene carbon is singly-bonded to two carbons, the other is singly-bonded to one carbon
            rxn2 = AllChem.ReactionFromSmarts('[C:6][C:1]([C:4])=[CH:2]([C:5]).[Cl:3]>>[C:6][C:1]([C:4])(-[Cl:3])[CH:2]([C:5])', useSmiles=True)
            ps2 = rxn2.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Cl]")))

            # both alkene carbons are each singly-bonded to two carbons 
            rxn3 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:2])=[C:3]([C:4])[C:5].[Cl:6]>>[C:0][C:1]([C:2])(-[Cl:6])[C:3]([C:4])[C:5]', useSmiles=True)
            ps3 = rxn3.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Cl]")))

            # one alkene carbon is singly-bonded to one carbon, the other is not bonded to any carbons
            rxn4 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH2:2].[Cl:3]>>[C:4][CH:1](-[Cl:3])-[CH2:2]', useSmiles=True)
            ps4 = rxn4.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Cl]")))

            # both alkene carbons each singly-bonded to one carbon
            rxn5 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH:2][C:5].[Cl:3]>>[C:4][CH:1](-[Cl:3])[CH:2][C:5]', useSmiles=True)
            ps5 = rxn5.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Cl]")))

            # both alkene carbons are not bonded to any carbons
            rxn6 = AllChem.ReactionFromSmarts('[CH2:1]=[CH2:2].[Cl:3]>>[CH2:1](-[Cl:3])[CH3:2]', useSmiles=True)
            ps6 = rxn6.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][Cl]")))

            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("[C][C]([C])=[CH2]")
            r1 = m1.HasSubstructMatch(patt1)

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[C][C]([C])=[CH][C]")
            r2 = m2.HasSubstructMatch(patt2)

            m3 = Chem.MolFromSmiles(smile1)
            patt3 = Chem.MolFromSmarts("[C][C]([C])=[C]([C])[C]")
            r3 = m3.HasSubstructMatch(patt3)

            m4 = Chem.MolFromSmiles(smile1)
            patt4 = Chem.MolFromSmarts("[C][CH]=[CH2]")
            r4 = m4.HasSubstructMatch(patt4)

            m5 = Chem.MolFromSmiles(smile1)
            patt5 = Chem.MolFromSmarts("[C][CH]=[CH][C]")
            r5 = m5.HasSubstructMatch(patt5)

            m6 = Chem.MolFromSmiles(smile1)
            patt6 = Chem.MolFromSmarts("[CH2]=[CH2]")
            r6 = m6.HasSubstructMatch(patt6)

            alkene_H_halide_reactions_list = [rxn1, rxn2, rxn3, rxn4, rxn5, rxn6] 
            H_halideaddition_products_list = [ps1, ps2, ps3, ps4, ps5, ps6]
            alkene_regiochemistry_list = [r1, r2, r3, r4, r5, r6]
            num_possible_products_list = [1, 1, 2, 1, 2, 1]
        
            i = 0
            for i in range(6):
                if alkene_regiochemistry_list[i] == True:
                    j = 0
                    for j in range(num_possible_products_list[i]):
                        smile_products.append(Chem.MolToSmiles(H_halideaddition_products_list[i][j][0]))
                        j += 1
                    del(smile_products[0])
                    smile1 = smile_products[0]
                    
                    i = 0
                    break
            
                i += 1

        return smile_products

    @staticmethod
    def HI_addition(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:

            # one alkene carbon has two carbons singly-bonded to it; the other alkene carbon is not bonded to any carbons
            rxn1 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:4])=[CH2:2].[I:3]>>[C:0][C:1]([C:4])(-[I:3])[CH2:2]', useSmiles=True)        
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][I]")))

            # one alkene carbon is singly-bonded to two carbons, the other is singly-bonded to one carbon
            rxn2 = AllChem.ReactionFromSmarts('[C:6][C:1]([C:4])=[CH:2]([C:5]).[I:3]>>[C:6][C:1]([C:4])(-[I:3])[CH:2]([C:5])', useSmiles=True)
            ps2 = rxn2.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][I]")))

            # both alkene carbons are each singly-bonded to two carbons 
            rxn3 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:2])=[C:3]([C:4])[C:5].[I:6]>>[C:0][C:1]([C:2])(-[I:6])[C:3]([C:4])[C:5]', useSmiles=True)
            ps3 = rxn3.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][I]")))

            # one alkene carbon is singly-bonded to one carbon, the other is not bonded to any carbons
            rxn4 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH2:2].[I:3]>>[C:4][CH:1](-[I:3])-[CH2:2]', useSmiles=True)
            ps4 = rxn4.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][I]")))

            # both alkene carbons each singly-bonded to one carbon
            rxn5 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH:2][C:5].[I:3]>>[C:4][CH:1](-[I:3])[CH:2][C:5]', useSmiles=True)
            ps5 = rxn5.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][I]")))

            # both alkene carbons are not bonded to any carbons
            rxn6 = AllChem.ReactionFromSmarts('[CH2:1]=[CH2:2].[I:3]>>[CH2:1](-[I:3])[CH3:2]', useSmiles=True)
            ps6 = rxn6.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][I]")))

            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("[C][C]([C])=[CH2]")
            r1 = m1.HasSubstructMatch(patt1)

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[C][C]([C])=[CH][C]")
            r2 = m2.HasSubstructMatch(patt2)

            m3 = Chem.MolFromSmiles(smile1)
            patt3 = Chem.MolFromSmarts("[C][C]([C])=[C]([C])[C]")
            r3 = m3.HasSubstructMatch(patt3)

            m4 = Chem.MolFromSmiles(smile1)
            patt4 = Chem.MolFromSmarts("[C][CH]=[CH2]")
            r4 = m4.HasSubstructMatch(patt4)

            m5 = Chem.MolFromSmiles(smile1)
            patt5 = Chem.MolFromSmarts("[C][CH]=[CH][C]")
            r5 = m5.HasSubstructMatch(patt5)

            m6 = Chem.MolFromSmiles(smile1)
            patt6 = Chem.MolFromSmarts("[CH2]=[CH2]")
            r6 = m6.HasSubstructMatch(patt6)

            alkene_H_halide_reactions_list = [rxn1, rxn2, rxn3, rxn4, rxn5, rxn6] 
            H_halideaddition_products_list = [ps1, ps2, ps3, ps4, ps5, ps6]
            alkene_regiochemistry_list = [r1, r2, r3, r4, r5, r6]
            num_possible_products_list = [1, 1, 2, 1, 2, 1]
        
            i = 0
            for i in range(6):
                if alkene_regiochemistry_list[i] == True:
                    j = 0
                    for j in range(num_possible_products_list[i]):
                        smile_products.append(Chem.MolToSmiles(H_halideaddition_products_list[i][j][0]))
                        j += 1
                    del(smile_products[0])
                    smile1 = smile_products[0]
                    
                    i = 0
                    break
            
                i += 1

        return smile_products

    @staticmethod
    def Br2_addition(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]=[C:2].[Br:3][Br:4]>>[C:1](-[Br:3])[C:2](-[Br:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("BrBr")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def Cl2_addition(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]=[C:2].[Cl:3][Cl:4]>>[C:1](-[Cl:3])[C:2](-[Cl:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("ClCl")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def bromohydrin_formation(smile): #the "-OH" will bond to the more-substituted carbon
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:

            # one alkene carbon has two carbons singly-bonded to it; the other alkene carbon is not bonded to any carbons
            rxn1 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:4])=[CH2:2].[O:8][Br:9]>>[C:0][C:1]([C:4])(-[O:8])[CH2:2](-[Br:9])', useSmiles=True)        
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("BrO")))

            # one alkene carbon is singly-bonded to two carbons, the other is singly-bonded to one carbon
            rxn2 = AllChem.ReactionFromSmarts('[C:6][C:1]([C:4])=[CH:2]([C:5]).[O:8][Br:9]>>[C:6][C:1]([C:4])(-[O:8])[CH:2](-[Br:9])([C:5])', useSmiles=True)
            ps2 = rxn2.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("BrO")))

            # both alkene carbons are each singly-bonded to two carbons 
            rxn3 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:2])=[C:3]([C:4])[C:5].[O:8][Br:9]>>[C:0][C:1]([C:2])(-[O:8])[C:3](-[Br:9])([C:4])[C:5]', useSmiles=True)
            ps3 = rxn3.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("BrO")))

            # one alkene carbon is singly-bonded to one carbon, the other is not bonded to any carbons
            rxn4 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH2:2].[O:8][Br:9]>>[C:4][CH:1](-[O:8])-[CH2:2](-[Br:9])', useSmiles=True)
            ps4 = rxn4.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("BrO")))

            # both alkene carbons each singly-bonded to one carbon
            rxn5 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH:2][C:5].[O:8][Br:9]>>[C:4][CH:1](-[O:8])[CH:2](-[Br:9])[C:5]', useSmiles=True)
            ps5 = rxn5.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("BrO")))

            # both alkene carbons are not bonded to any carbons
            rxn6 = AllChem.ReactionFromSmarts('[CH2:1]=[CH2:2].[O:8][Br:9]>>[CH2:1](-[O:8])[CH3:2](-[Br:9])', useSmiles=True)
            ps6 = rxn6.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("BrO")))

            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("[C][C]([C])=[CH2]")
            r1 = m1.HasSubstructMatch(patt1)

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[C][C]([C])=[CH][C]")
            r2 = m2.HasSubstructMatch(patt2)

            m3 = Chem.MolFromSmiles(smile1)
            patt3 = Chem.MolFromSmarts("[C][C]([C])=[C]([C])[C]")
            r3 = m3.HasSubstructMatch(patt3)

            m4 = Chem.MolFromSmiles(smile1)
            patt4 = Chem.MolFromSmarts("[C][CH]=[CH2]")
            r4 = m4.HasSubstructMatch(patt4)

            m5 = Chem.MolFromSmiles(smile1)
            patt5 = Chem.MolFromSmarts("[C][CH]=[CH][C]")
            r5 = m5.HasSubstructMatch(patt5)

            m6 = Chem.MolFromSmiles(smile1)
            patt6 = Chem.MolFromSmarts("[CH2]=[CH2]")
            r6 = m6.HasSubstructMatch(patt6)

            alkene_H_halide_reactions_list = [rxn1, rxn2, rxn3, rxn4, rxn5, rxn6] 
            H_halideaddition_products_list = [ps1, ps2, ps3, ps4, ps5, ps6]
            alkene_regiochemistry_list = [r1, r2, r3, r4, r5, r6]
            num_possible_products_list = [1, 1, 2, 1, 2, 1]
        
            i = 0
            for i in range(6):
                if alkene_regiochemistry_list[i] == True:
                    j = 0
                    for j in range(num_possible_products_list[i]):
                        smile_products.append(Chem.MolToSmiles(H_halideaddition_products_list[i][j][0]))
                        j += 1
                    del(smile_products[0])
                    smile1 = smile_products[0]
                    
                    i = 0
                    break
            
                i += 1

        return smile_products

    @staticmethod
    def chlorohydrin_formation(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:

            # one alkene carbon has two carbons singly-bonded to it; the other alkene carbon is not bonded to any carbons
            rxn1 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:4])=[CH2:2].[O:8][Cl:9]>>[C:0][C:1]([C:4])(-[O:8])[CH2:2](-[Cl:9])', useSmiles=True)        
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("ClO")))

            # one alkene carbon is singly-bonded to two carbons, the other is singly-bonded to one carbon
            rxn2 = AllChem.ReactionFromSmarts('[C:6][C:1]([C:4])=[CH:2]([C:5]).[O:8][Cl:9]>>[C:6][C:1]([C:4])(-[O:8])[CH:2](-[Cl:9])([C:5])', useSmiles=True)
            ps2 = rxn2.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("ClO")))

            # both alkene carbons are each singly-bonded to two carbons 
            rxn3 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:2])=[C:3]([C:4])[C:5].[O:8][Cl:9]>>[C:0][C:1]([C:2])(-[O:8])[C:3](-[Cl:9])([C:4])[C:5]', useSmiles=True)
            ps3 = rxn3.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("ClO")))

            # one alkene carbon is singly-bonded to one carbon, the other is not bonded to any carbons
            rxn4 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH2:2].[O:8][Cl:9]>>[C:4][CH:1](-[O:8])-[CH2:2](-[Cl:9])', useSmiles=True)
            ps4 = rxn4.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("ClO")))

            # both alkene carbons each singly-bonded to one carbon
            rxn5 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH:2][C:5].[O:8][Cl:9]>>[C:4][CH:1](-[O:8])[CH:2](-[Cl:9])[C:5]', useSmiles=True)
            ps5 = rxn5.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("ClO")))

            # both alkene carbons are not bonded to any carbons
            rxn6 = AllChem.ReactionFromSmarts('[CH2:1]=[CH2:2].[O:8][Cl:9]>>[CH2:1](-[O:8])[CH3:2](-[Cl:9])', useSmiles=True)
            ps6 = rxn6.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("ClO")))

            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("[C][C]([C])=[CH2]")
            r1 = m1.HasSubstructMatch(patt1)

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[C][C]([C])=[CH][C]")
            r2 = m2.HasSubstructMatch(patt2)

            m3 = Chem.MolFromSmiles(smile1)
            patt3 = Chem.MolFromSmarts("[C][C]([C])=[C]([C])[C]")
            r3 = m3.HasSubstructMatch(patt3)

            m4 = Chem.MolFromSmiles(smile1)
            patt4 = Chem.MolFromSmarts("[C][CH]=[CH2]")
            r4 = m4.HasSubstructMatch(patt4)

            m5 = Chem.MolFromSmiles(smile1)
            patt5 = Chem.MolFromSmarts("[C][CH]=[CH][C]")
            r5 = m5.HasSubstructMatch(patt5)

            m6 = Chem.MolFromSmiles(smile1)
            patt6 = Chem.MolFromSmarts("[CH2]=[CH2]")
            r6 = m6.HasSubstructMatch(patt6)

            alkene_H_halide_reactions_list = [rxn1, rxn2, rxn3, rxn4, rxn5, rxn6] 
            H_halideaddition_products_list = [ps1, ps2, ps3, ps4, ps5, ps6]
            alkene_regiochemistry_list = [r1, r2, r3, r4, r5, r6]
            num_possible_products_list = [1, 1, 2, 1, 2, 1]
        
            i = 0
            for i in range(6):
                if alkene_regiochemistry_list[i] == True:
                    j = 0
                    for j in range(num_possible_products_list[i]):
                        smile_products.append(Chem.MolToSmiles(H_halideaddition_products_list[i][j][0]))
                        j += 1
                    del(smile_products[0])
                    smile1 = smile_products[0]
                    
                    i = 0
                    break
            
                i += 1

        return smile_products

    @staticmethod
    def iodohydrin_formation(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:

            # one alkene carbon has two carbons singly-bonded to it; the other alkene carbon is not bonded to any carbons
            rxn1 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:4])=[CH2:2].[O:8][I:9]>>[C:0][C:1]([C:4])(-[O:8])[CH2:2](-[I:9])', useSmiles=True)        
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("IO")))

            # one alkene carbon is singly-bonded to two carbons, the other is singly-bonded to one carbon
            rxn2 = AllChem.ReactionFromSmarts('[C:6][C:1]([C:4])=[CH:2]([C:5]).[O:8][I:9]>>[C:6][C:1]([C:4])(-[O:8])[CH:2](-[I:9])([C:5])', useSmiles=True)
            ps2 = rxn2.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("IO")))

            # both alkene carbons are each singly-bonded to two carbons 
            rxn3 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:2])=[C:3]([C:4])[C:5].[O:8][I:9]>>[C:0][C:1]([C:2])(-[O:8])[C:3](-[I:9])([C:4])[C:5]', useSmiles=True)
            ps3 = rxn3.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("IO")))

            # one alkene carbon is singly-bonded to one carbon, the other is not bonded to any carbons
            rxn4 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH2:2].[O:8][I:9]>>[C:4][CH:1](-[O:8])-[CH2:2](-[I:9])', useSmiles=True)
            ps4 = rxn4.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("IO")))

            # both alkene carbons each singly-bonded to one carbon
            rxn5 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH:2][C:5].[O:8][I:9]>>[C:4][CH:1](-[O:8])[CH:2](-[I:9])[C:5]', useSmiles=True)
            ps5 = rxn5.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("IO")))

            # both alkene carbons are not bonded to any carbons
            rxn6 = AllChem.ReactionFromSmarts('[CH2:1]=[CH2:2].[O:8][I:9]>>[CH2:1](-[O:8])[CH3:2](-[I:9])', useSmiles=True)
            ps6 = rxn6.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("IO")))

            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("[C][C]([C])=[CH2]")
            r1 = m1.HasSubstructMatch(patt1)

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[C][C]([C])=[CH][C]")
            r2 = m2.HasSubstructMatch(patt2)

            m3 = Chem.MolFromSmiles(smile1)
            patt3 = Chem.MolFromSmarts("[C][C]([C])=[C]([C])[C]")
            r3 = m3.HasSubstructMatch(patt3)

            m4 = Chem.MolFromSmiles(smile1)
            patt4 = Chem.MolFromSmarts("[C][CH]=[CH2]")
            r4 = m4.HasSubstructMatch(patt4)

            m5 = Chem.MolFromSmiles(smile1)
            patt5 = Chem.MolFromSmarts("[C][CH]=[CH][C]")
            r5 = m5.HasSubstructMatch(patt5)

            m6 = Chem.MolFromSmiles(smile1)
            patt6 = Chem.MolFromSmarts("[CH2]=[CH2]")
            r6 = m6.HasSubstructMatch(patt6)

            alkene_H_halide_reactions_list = [rxn1, rxn2, rxn3, rxn4, rxn5, rxn6] 
            H_halideaddition_products_list = [ps1, ps2, ps3, ps4, ps5, ps6]
            alkene_regiochemistry_list = [r1, r2, r3, r4, r5, r6]
            num_possible_products_list = [1, 1, 2, 1, 2, 1]
        
            i = 0
            for i in range(6):
                if alkene_regiochemistry_list[i] == True:
                    j = 0
                    for j in range(num_possible_products_list[i]):
                        smile_products.append(Chem.MolToSmiles(H_halideaddition_products_list[i][j][0]))
                        j += 1
                    del(smile_products[0])
                    smile1 = smile_products[0]
                    
                    i = 0
                    break
            
                i += 1

        return smile_products

    @staticmethod
    def Markovnikov_water_addition(smile):
        smile_products = [smile] # following Markovnikov's rule, the "-OH" will ad to the more-substituted carbon
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
            # one alkene carbon has two carbons singly-bonded to it; the other alkene carbon is not bonded to any carbons
            rxn1 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:4])=[CH2:2].[O:3]>>[C:0][C:1]([C:4])(-[O:3])[CH2:2]', useSmiles=True)        
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            # one alkene carbon is singly-bonded to two carbons, the other is singly-bonded to one carbon
            rxn2 = AllChem.ReactionFromSmarts('[C:6][C:1]([C:4])=[CH:2]([C:5]).[O:3]>>[C:6][C:1]([C:4])(-[O:3])[CH:2]([C:5])', useSmiles=True)
            ps2 = rxn2.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            # both alkene carbons are each singly-bonded to two carbons 
            rxn3 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:2])=[C:3]([C:4])[C:5].[O:6]>>[C:0][C:1]([C:2])(-[O:6])[C:3]([C:4])[C:5]', useSmiles=True)
            ps3 = rxn3.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            # one alkene carbon is singly-bonded to one carbon, the other is not bonded to any carbons
            rxn4 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH2:2].[O:3]>>[C:4][CH:1](-[O:3])-[CH2:2]', useSmiles=True)
            ps4 = rxn4.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            # both alkene carbons each singly-bonded to one carbon
            rxn5 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH:2][C:5].[O:3]>>[C:4][CH:1](-[O:3])[CH:2][C:5]', useSmiles=True)
            ps5 = rxn5.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            # both alkene carbons are not bonded to any carbons
            rxn6 = AllChem.ReactionFromSmarts('[CH2:1]=[CH2:2].[O:3]>>[CH2:1](-[O:3])[CH3:2]', useSmiles=True)
            ps6 = rxn6.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("[C][C]([C])=[CH2]")
            r1 = m1.HasSubstructMatch(patt1)

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[C][C]([C])=[CH][C]")
            r2 = m2.HasSubstructMatch(patt2)

            m3 = Chem.MolFromSmiles(smile1)
            patt3 = Chem.MolFromSmarts("[C][C]([C])=[C]([C])[C]")
            r3 = m3.HasSubstructMatch(patt3)

            m4 = Chem.MolFromSmiles(smile1)
            patt4 = Chem.MolFromSmarts("[C][CH]=[CH2]")
            r4 = m4.HasSubstructMatch(patt4)

            m5 = Chem.MolFromSmiles(smile1)
            patt5 = Chem.MolFromSmarts("[C][CH]=[CH][C]")
            r5 = m5.HasSubstructMatch(patt5)

            m6 = Chem.MolFromSmiles(smile1)
            patt6 = Chem.MolFromSmarts("[CH2]=[CH2]")
            r6 = m6.HasSubstructMatch(patt6)

            alkene_H_halide_reactions_list = [rxn1, rxn2, rxn3, rxn4, rxn5, rxn6] 
            H_halideaddition_products_list = [ps1, ps2, ps3, ps4, ps5, ps6]
            alkene_regiochemistry_list = [r1, r2, r3, r4, r5, r6]
            num_possible_products_list = [1, 1, 2, 1, 2, 1]
        
            i = 0
            for i in range(6):
                if alkene_regiochemistry_list[i] == True:
                    j = 0
                    for j in range(num_possible_products_list[i]):
                        smile_products.append(Chem.MolToSmiles(H_halideaddition_products_list[i][j][0]))
                        j += 1
                    del(smile_products[0])
                    smile1 = smile_products[0]
                    
                    i = 0
                    break
            
                i += 1

        return smile_products

    @staticmethod
    def nonMarkovnikov_water_addition(smile):
        smile_products = [smile] # the "-OH" will add to the less-substituted carbon
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
            # one alkene carbon has two carbons singly-bonded to it; the other alkene carbon is not bonded to any carbons
            rxn1 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:4])=[CH2:2].[O:3]>>[C:0][C:1]([C:4])[CH2:2](-[O:3])', useSmiles=True)        
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            # one alkene carbon is singly-bonded to two carbons, the other is singly-bonded to one carbon
            rxn2 = AllChem.ReactionFromSmarts('[C:6][C:1]([C:4])=[CH:2]([C:5]).[O:3]>>[C:6][C:1]([C:4])[CH:2](-[O:3])([C:5])', useSmiles=True)
            ps2 = rxn2.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            # both alkene carbons are each singly-bonded to two carbons 
            rxn3 = AllChem.ReactionFromSmarts('[C:0][C:1]([C:2])=[C:3]([C:4])[C:5].[O:6]>>[C:0][C:1]([C:2])[C:3](-[O:6])([C:4])[C:5]', useSmiles=True)
            ps3 = rxn3.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            # one alkene carbon is singly-bonded to one carbon, the other is not bonded to any carbons
            rxn4 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH2:2].[O:3]>>[C:4][CH:1]-[CH2:2](-[O:3])', useSmiles=True)
            ps4 = rxn4.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            # both alkene carbons each singly-bonded to one carbon
            rxn5 = AllChem.ReactionFromSmarts('[C:4][CH:1]=[CH:2][C:5].[O:3]>>[C:4][CH:1][CH:2](-[O:3])[C:5]', useSmiles=True)
            ps5 = rxn5.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            # both alkene carbons are not bonded to any carbons
            rxn6 = AllChem.ReactionFromSmarts('[CH2:1]=[CH2:2].[O:3]>>[CH2:1](-[O:3])[CH3:2]', useSmiles=True)
            ps6 = rxn6.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H]O[H]")))

            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("[C][C]([C])=[CH2]")
            r1 = m1.HasSubstructMatch(patt1)

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[C][C]([C])=[CH][C]")
            r2 = m2.HasSubstructMatch(patt2)

            m3 = Chem.MolFromSmiles(smile1)
            patt3 = Chem.MolFromSmarts("[C][C]([C])=[C]([C])[C]")
            r3 = m3.HasSubstructMatch(patt3)

            m4 = Chem.MolFromSmiles(smile1)
            patt4 = Chem.MolFromSmarts("[C][CH]=[CH2]")
            r4 = m4.HasSubstructMatch(patt4)

            m5 = Chem.MolFromSmiles(smile1)
            patt5 = Chem.MolFromSmarts("[C][CH]=[CH][C]")
            r5 = m5.HasSubstructMatch(patt5)

            m6 = Chem.MolFromSmiles(smile1)
            patt6 = Chem.MolFromSmarts("[CH2]=[CH2]")
            r6 = m6.HasSubstructMatch(patt6)

            alkene_H_halide_reactions_list = [rxn1, rxn2, rxn3, rxn4, rxn5, rxn6] 
            H_halideaddition_products_list = [ps1, ps2, ps3, ps4, ps5, ps6]
            alkene_regiochemistry_list = [r1, r2, r3, r4, r5, r6]
            num_possible_products_list = [1, 1, 2, 1, 2, 1]
        
            i = 0
            for i in range(6):
                if alkene_regiochemistry_list[i] == True:
                    j = 0
                    for j in range(num_possible_products_list[i]):
                        smile_products.append(Chem.MolToSmiles(H_halideaddition_products_list[i][j][0]))
                        j += 1
                    del(smile_products[0])
                    smile1 = smile_products[0]
                    
                    i = 0
                    break
            
                i += 1

        return smile_products

    @staticmethod
    def catalytic_hydrogenation(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]=[C:2].[H:3][H:4]>>[C:1](-[H:3])[C:2](-[H:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][H]")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def epoxidation(smile): 
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]=[C:2].[O:3]>>[C:1]1[C:2][O:3]1', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("O")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products


    @staticmethod
    def hydroxylation(smile): 
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.epoxide(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]1[C:2][O:3]1.[O:4]>>[C:1]([O:3])[C:2]([O:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("O")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def dichlorocyclopropane_formation(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]=[C:2].[Cl:3][C:4][Cl:5]>>[Cl:3][C:4]1([Cl:5])[C:1][C:2]1', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("ClCCl")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def dibromocyclopropane_formation(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]=[C:2].[Br:3][C:4][Br:5]>>[Br:3][C:4]1([Br:5])[C:1][C:2]1', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("BrCBr")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def diiodocyclopropane_formation(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]=[C:2].[I:3][C:4][I:5]>>[I:3][C:4]1([I:5])[C:1][C:2]1', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("ICI")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def Simmons_Smith_reaction(smile):
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.alkene(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]=[C:2].[C:3]>>[C:1]1[C:2][C:3]1', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("C")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def alkene_ozonation(smile):
        smile_products = [smile]

        i = 0
        while i in range(len(smile_products)):
            if FunctionalGroups.ring_alkene(smile_products[i]) == True: 
                # if alkene is in ring, ensure that the cleavage results in one molecule, not two
                m = Chem.MolFromSmiles(smile_products[i])

                bis = m.GetSubstructMatches(Chem.MolFromSmarts('[C;R]=[C;R]'))
                bs = [m.GetBondBetweenAtoms(x,y).GetIdx() for x,y in bis]
                bs = []
                labels=[]
                for bi in bis:
                    b = m.GetBondBetweenAtoms(bi[0],bi[1])
                    if b.GetBeginAtomIdx()==bi[0]:
                        labels.append((100,100))
                    else:
                        labels.append((100,100))
                    bs.append(b.GetIdx())

                nm = Chem.FragmentOnBonds(m,bs,dummyLabels=labels)
                smile_products.append(Chem.MolToSmiles(nm,True))
                del(smile_products[i])

                while FunctionalGroups.cleaved_alkene(smile_products[i]) == True:
                    rxn1 = AllChem.ReactionFromSmarts('[C:1]=[100*:2].[O:3]>>[C:1](=[O:3]).[100*:2]', useSmiles=True)
                    ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("O")))
                    smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                    del(smile_products[i])
                    i = i
                
            elif FunctionalGroups.alkene(smile_products[i]) == True:
                # alkene cleavage should result in two molecules
                rxn1 = AllChem.ReactionFromSmarts('[C:1]=[C:2].[O:3][O:4]>>[C:1]=[O:3].[C:2]=[O:4]', useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("[O][O]")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                smile_products.append(Chem.MolToSmiles(ps1[0][1]))
                del(smile_products[i])
                i = 0
                
            else:
                i += 1
                    
        return smile_products

    @staticmethod
    def KMnO4_alkene_cleavage(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        
        i = 0
        while i in range(len(smile_products)):
            if FunctionalGroups.ring_alkene(smile_products[i]) == True:
                m = Chem.MolFromSmiles(smile_products[i])
                bis = m.GetSubstructMatches(Chem.MolFromSmarts('[C;R]=[C;R]'))
                bs = [m.GetBondBetweenAtoms(x,y).GetIdx() for x,y in bis]
                bs = []
                labels=[]
                for bi in bis:
                    b = m.GetBondBetweenAtoms(bi[0],bi[1])
                    if b.GetBeginAtomIdx()==bi[0]:
                        labels.append((100,100))
                    else:
                        labels.append((100,100))
                    bs.append(b.GetIdx())

                nm = Chem.FragmentOnBonds(m,bs,dummyLabels=labels)
                smile_products.append(Chem.MolToSmiles(nm,True))
                print(Chem.MolToSmiles(nm,True))
                del(smile_products[i])
                
                while FunctionalGroups.cleaved_alkene(smile_products[i]) == True:
                    # One hydrogen bonded to the carbon
                    rxn1 = AllChem.ReactionFromSmarts('[C:0][CH:1]=[100*:2].[O:3]=[O:4]>>[C:0][C:1](=[O:3])-[O:4].[100*:2]', useSmiles=True)
                    ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("O=O")))

                    # No hydrogens bonded to the carbon
                    rxn2 = AllChem.ReactionFromSmarts('[C:1]=[100*:2].[O:3]>>[C:1]=[O:3].[100*:2]', useSmiles=True)
                    ps2 = rxn2.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("O")))

                    m1 = Chem.MolFromSmiles(smile_products[i])
                    patt1 = Chem.MolFromSmarts("[CH]=[100*]")
                    r1 = m1.HasSubstructMatch(patt1)

                    m2 = Chem.MolFromSmiles(smile_products[i])
                    patt2 = Chem.MolFromSmarts("[CH0](=[100*])")
                    r2 = m2.HasSubstructMatch(patt2)    

                    if r1 == True:
                        smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                        del(smile_products[i])
                        i = 0
                    elif r2 == True:
                        smile_products.append(Chem.MolToSmiles(ps2[0][0]))
                        del(smile_products[i])
                        i = 0
                
            elif FunctionalGroups.nonring_alkene(smile_products[i]) == True:
                    
                # Two hydrogens bonded to the carbon
                rxn1 = AllChem.ReactionFromSmarts('[CH2:1]=[CH2:2].([O:3]=[O:4].[O:5]=[O:6])>>[O:3]=[C:1]=[O:4].[O:5]=[C:2]=[O:6]', useSmiles=True)       
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("O=O.O=O")))

                # One hydrogen bonded to the carbon
                rxn2 = AllChem.ReactionFromSmarts('[CH2:1]=[CH:2].([O:3]=[O:4].[O:5]=[O:6])>>[O:3]=[C:1]=[O:4].[C:2]([O:5])=[O:6]', useSmiles=True)
                ps2 = rxn2.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("O=O.O=O")))

                # No hydrogens bonded to the carbon
                rxn3 = AllChem.ReactionFromSmarts('[CH2:1]=[CH0:2].([O:3]=[O:4].[O:5])>>[O:3]=[C:1]=[O:4].[CH0:2]=[O:5]', useSmiles=True)
                ps3 = rxn3.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("O=O.O")))

                rxn4 = AllChem.ReactionFromSmarts('[CH:1]=[CH:2].([O:3]=[O:4].[O:5]=[O:6])>>[C:1]([O:3])=[O:4].[C:2]([O:5])=[O:6]', useSmiles=True)
                ps4 = rxn4.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("O=O.O=O")))

                rxn5 = AllChem.ReactionFromSmarts('[CH:1]=[CH0:2].([O:3]=[O:4].[O:5])>>[C:1]([O:3])=[O:4].[CH0:2]=[O:5]', useSmiles=True)
                ps5 = rxn5.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("O=O.O")))

                rxn6 = AllChem.ReactionFromSmarts('[CH0:1]=[CH0:2].[O:3]=[O:4]>>[CH0:1]=[O:3].[CH0:2]=[O:4]', useSmiles=True)
                ps6 = rxn6.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("O=O")))

                m1 = Chem.MolFromSmiles(smile_products[i])
                patt1 = Chem.MolFromSmarts("[CH2]=[CH2]")
                r1 = m1.HasSubstructMatch(patt1)

                m2 = Chem.MolFromSmiles(smile_products[i])
                patt2 = Chem.MolFromSmarts("[CH2]=[CH]")
                r2 = m2.HasSubstructMatch(patt2)

                m3 = Chem.MolFromSmiles(smile_products[i])
                patt3 = Chem.MolFromSmarts("[CH2]=[CH0]")
                r3 = m3.HasSubstructMatch(patt3)    
                            
                m4 = Chem.MolFromSmiles(smile_products[i])
                patt4 = Chem.MolFromSmarts("[CH]=[CH]")
                r4 = m4.HasSubstructMatch(patt4) 

                m5 = Chem.MolFromSmiles(smile_products[i])
                patt5 = Chem.MolFromSmarts("[CH]=[CH0]")
                r5 = m5.HasSubstructMatch(patt5) 

                m6 = Chem.MolFromSmiles(smile_products[i])
                patt6 = Chem.MolFromSmarts("[CH0]=[CH0]")
                r6 = m6.HasSubstructMatch(patt6) 

                list_of_possible_configurations = [r1, r2, r3, r4, r5, r6]
                list_of_possible_reactions = [ps1, ps2, ps3, ps4, ps5, ps6]

                j = 0
                for j in range(6):
                    if list_of_possible_configurations[j] == True:
                        smile_products.append(Chem.MolToSmiles(list_of_possible_reactions[j][0][0]))
                        smile_products.append(Chem.MolToSmiles(list_of_possible_reactions[j][0][1]))

                        del(smile_products[i])
                        i = i
                    
                    else:
                        j += 1
            else:
                i += 1

        return smile_products

    @staticmethod
    def oxidative_cleavage_of_1_2_diols(smile):
        smile_products = [smile]
        i = 0

        while i in range(len(smile_products)):
            if FunctionalGroups.ring_1_2_diols(smile_products[i]) == True:
                
                m = Chem.MolFromSmiles(smile_products[i])
                bis = m.GetSubstructMatches(Chem.MolFromSmarts('[C;R]([O])[C;R]([O])')) 
                bs = [m.GetBondBetweenAtoms(a,c).GetIdx() for a,b,c,d in bis]
                bs = []
                labels=[]
                for bi in bis:
                    b = m.GetBondBetweenAtoms(bi[0],bi[2])
                    if b.GetBeginAtomIdx()==bi[0]:
                        labels.append((100,100))
                    else:
                        labels.append((100,100))
                    bs.append(b.GetIdx())

                nm = Chem.FragmentOnBonds(m,bs,dummyLabels=labels)
                smile_products.append(Chem.MolToSmiles(nm,True))
                del(smile_products[i])
                

                while FunctionalGroups.cleaved_ring_vicinal_diols(smile_products[i]) == True:
                    rxn1 = AllChem.ReactionFromSmarts('[C:1]([100*:2])[O:3].[O:4]>>[C:1]=[O:4].[100*:2].[O:3]', useSmiles=True)
                    ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("O")))
                    smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                    del(smile_products[i])
                    i = i
                
            elif FunctionalGroups.non_ring_1_2_diols(smile_products[i]) == True:
                rxn1 = AllChem.ReactionFromSmarts('[C:1]([O:2])-[C:3]([O:4]).[O:5][O:6]>>[C:1]=[O:5].[C:3]=[O:6].[O:2][O:4]', useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile_products[i]),Chem.MolFromSmiles("[O][O]")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                smile_products.append(Chem.MolToSmiles(ps1[0][1]))
                del(smile_products[i])
                i = 0
                
            else:
                i += 1
        
        return smile_products
            

    @staticmethod
    def partial_alkyne_hydrogenation(smile):
        smile_products = [smile]
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile1 = smile_products[0]
        
        while FunctionalGroups.terminal_alkyne(smile1) == True or FunctionalGroups.internal_alkyne(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[H:3][H:4]>>[C:1](-[H:3])=[C:2](-[H:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][H]")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def terminal_alkyne_alkane_addition(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.terminal_alkyne(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts("[C:1]#[CH:2].[C:4][C:5]>>[C:4][C:1]#[C:2].[C:5]", useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("CC")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products


    @staticmethod
    def getAllReactions():
        method_list = [eval("AlkeneReactions." + method) for method in dir(AlkeneReactions) if method.startswith('_') is False and method.startswith("getAllReactions") is False]
        return method_list

# AlkeneReactions.terminal_alkyne_methane_addition("C#CCCCCC#CC")
AlkeneReactions.partial_alkyne_hydrogenation("C#CCC#CC")
# AlkeneReactions.partial_alkyne_hydrogenation("C#C")

def evaluate1(smile1, smile2):
    m1 = Chem.MolFromSmiles(smile1)
    fp1 = AllChem.GetMorganFingerprint(m1,3)
    m2 = Chem.MolFromSmiles(smile2)
    fp2 = AllChem.GetMorganFingerprint(m2,3)
    # DataStructs.DiceSimilarity(fp1,fp2)
    return DataStructs.DiceSimilarity(fp1, fp2)


def SHITTS(initialSMILE, finalSMILE):
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
    print("Search completed in", f'{time.time() - t:.5f} seconds')
    print("The reactions pathway is", reactions_pathway)
    print("The products formed are", products_pathway)

SHITTS("CCC#C", "CC=O")

def phosphine_oxidation(smile):
    smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
    smile_products = [smile]
    smile1 = smile_products[0]
    
    m1 = Chem.MolFromSmiles(smile1)
    patt1 = Chem.MolFromSmarts("[P;X3]")
    patt2 = Chem.MolFromSmarts("[P;X4]")

    while m1.HasSubstructMatch(patt1) == True and m1.HasSubstructMatch(patt2)  == False:
        rxn1 = AllChem.ReactionFromSmarts("[P:1].[O:2]>>[P:1](=[O:2])", useSmiles=True)
        ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("O")))
        smile_products.append(Chem.MolToSmiles(ps1[0][0]))
        del(smile_products[0])
        smile1 = smile_products[0]
        m1 = Chem.MolFromSmiles(smile1)

    return smile_products

phosphine_oxidation("CP(CCCCC)C")