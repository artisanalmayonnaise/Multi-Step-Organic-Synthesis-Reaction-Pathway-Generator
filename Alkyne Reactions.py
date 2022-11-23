import pandas as pd
import numpy as np
import random
import threading
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

    @staticmethod
    def vicinal_dihalides(smile):
        m = Chem.MolFromSmiles(smile)
        patt = Chem.MolFromSmarts("C([Br,Cl,F,I])C([Br,Cl,F,I])")
        FG_found = m.HasSubstructMatch(patt)

        if FG_found == True:
            return True
        else:
            return False

class AlkyneReactions:

    """
    The reactions contained in this class:
    - alkyne reduction to alkene: Lindlar catalyst and lithium in ammonia
    - dehydrohalogenation of vicinal dihalides
    - alkylation of acetylide anions
    - HCl and HBr and HI addition to alkynes
    - Br2 and Cl2 addition to alkynes
    - mercuric sulfate catalyzed ketone formation
    - hydroboration-oxidation: aldehyde formation
    - catalytic hydrogenation
    - conversion into acetylide anions
    - alkylation of acetylide anions

    """

    @staticmethod
    def partial_alkyne_hydrogenation(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]
        
        while FunctionalGroups.terminal_alkyne(smile1) == True or FunctionalGroups.internal_alkyne(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[H:3][H:4]>>[C:1](-[H:3])=[C:2](-[H:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][H]")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def vicinal_dehydrohalogenation(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]

        while FunctionalGroups.vicinal_dihalides(smile1) == True:
            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("C(Br)C(Br)")

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("C(Cl)C(Cl)")

            m3 = Chem.MolFromSmiles(smile1)
            patt3 = Chem.MolFromSmarts("C(I)C(I)")

            m4 = Chem.MolFromSmiles(smile1)
            patt4 = Chem.MolFromSmarts("C(F)C(F)")

            if m1.HasSubstructMatch(patt1) == True:
                rxn1 = AllChem.ReactionFromSmarts("[C:1]([Br:2])[C:3]([Br:4]).[H:5][H:6]>>[C:1]#[C:3].[Br:2][H:5].[Br:4][H:6]", useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1), Chem.MolFromSmiles("[H][H]")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                del(smile_products[0])
                smile1 = smile_products[0]


            if m2.HasSubstructMatch(patt2) == True:
                rxn1 = AllChem.ReactionFromSmarts("[C:1]([Cl:2])[C:3]([Cl:4]).[H:5][H:6]>>[C:1]#[C:3].[Cl:2][H:5].[Cl:4][H:6]", useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1), Chem.MolFromSmiles("[H][H]")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                del(smile_products[0])
                smile1 = smile_products[0]


            if m3.HasSubstructMatch(patt3) == True:
                rxn1 = AllChem.ReactionFromSmarts("[C:1]([I:2])[C:3]([I:4]).[H:5][H:6]>>[C:1]#[C:3].[I:2][H:5].[I:4][H:6]", useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1), Chem.MolFromSmiles("[H][H]")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                del(smile_products[0])
                smile1 = smile_products[0]

            if m4.HasSubstructMatch(patt4) == True:
                rxn1 = AllChem.ReactionFromSmarts("[C:1]([F:2])[C:3]([F:4]).[H:5][H:6]>>[C:1]#[C:3].[F:2][H:5].[F:4][H:6]", useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1), Chem.MolFromSmiles("[H][H]")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                del(smile_products[0])
                smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def catalytic_hydrogenation(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]

        while FunctionalGroups.terminal_alkyne(smile1)==True or FunctionalGroups.internal_alkyne(smile1)==True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[H:3][H:4]>>[C:1](-[H:3])[C:2](-[H:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][H]")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def HBr_addition(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]

        while FunctionalGroups.terminal_alkyne(smile1)==True or FunctionalGroups.internal_alkyne(smile1)==True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[Br:4]>>[C:1]=[C:2]([Br:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[Br]")))

            for i in range(2):
                smile_products.append(Chem.MolToSmiles(ps1[i][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def HCl_addition(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]

        while FunctionalGroups.terminal_alkyne(smile1)==True or FunctionalGroups.internal_alkyne(smile1)==True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[Cl:4]>>[C:1]=[C:2]([Cl:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[Cl]")))

            for i in range(2):
                smile_products.append(Chem.MolToSmiles(ps1[i][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def HI_addition(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]

        while FunctionalGroups.terminal_alkyne(smile1)==True or FunctionalGroups.internal_alkyne(smile1)==True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[I:4]>>[C:1]=[C:2]([I:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[I]")))

            for i in range(2):
                smile_products.append(Chem.MolToSmiles(ps1[i][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products


    @staticmethod
    def Br2_addition(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]

        while FunctionalGroups.terminal_alkyne(smile1)==True or FunctionalGroups.internal_alkyne(smile1)==True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[Br:3][Br:4]>>[C:1]([Br:3])=[C:2]([Br:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[Br][Br]")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def Cl2_addition(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]

        while FunctionalGroups.terminal_alkyne(smile1)==True or FunctionalGroups.internal_alkyne(smile1)==True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[Cl:3][Cl:4]>>[C:1]([Cl:3])=[C:2]([Cl:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[Cl][Cl]")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def I2_addition(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]

        while FunctionalGroups.terminal_alkyne(smile1)==True or FunctionalGroups.internal_alkyne(smile1)==True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[I:3][I:4]>>[C:1]([I:3])=[C:2]([I:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[I][I]")))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def mercuric_sulfate_catalyzed_ketone_formation(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]

        while FunctionalGroups.terminal_alkyne(smile1)==True:

            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("CC#[CH]")

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[CH]#[CH]")

            
            if m1.HasSubstructMatch(patt1) == True:
                rxn1 = AllChem.ReactionFromSmarts('[C:1]#[CH1:2].[O:3]>>[C:0][C:1](=[O:3])-[CH1:2]', useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[O]")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                del(smile_products[0])
                smile1 = smile_products[0]

            if m2.HasSubstructMatch(patt2) == True:
                rxn1 = AllChem.ReactionFromSmarts('[CH1:1]#[CH1:2].[O:3]>>[CH:1](=[O:3])[CH1:2]', useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("O")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                del(smile_products[0])
                smile1 = smile_products[0]


        return smile_products

    @staticmethod
    def mercuric_sulfate_catalyzed_enol_formation(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]

        while FunctionalGroups.terminal_alkyne(smile1)==True:
            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("CC#[CH]")

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[CH]#[CH]")

            if m1.HasSubstructMatch(patt1) == True:
                rxn1 = AllChem.ReactionFromSmarts('[C:0][C:1]#[CH1:2].[O:3]>>[C:0][C:1](-[O:3])=[CH1:2]', useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("O")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                del(smile_products[0])
                smile1 = smile_products[0]

            if m2.HasSubstructMatch(patt2) == True:
                rxn1 = AllChem.ReactionFromSmarts('[CH1:1]#[CH1:2].[O:3]>>[CH:1](-[O:3])=[CH1:2]', useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("O")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                del(smile_products[0])
                smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def hydroboration_oxidation(smile):
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile_products = [smile]
        smile1 = smile_products[0]
        # print(smile1)

        while FunctionalGroups.terminal_alkyne(smile1)==True:
            m1 = Chem.MolFromSmiles(smile1)
            patt1 = Chem.MolFromSmarts("CC#[CH]")

            m2 = Chem.MolFromSmiles(smile1)
            patt2 = Chem.MolFromSmarts("[CH]#[CH]")

            if m1.HasSubstructMatch(patt1) == True:
                rxn1 = AllChem.ReactionFromSmarts('[C:0][C:1]#[CH1:2].[O:3]>>[C:0][C:1][C:2](=[O:3])', useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("O")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                del(smile_products[0])
                smile1 = smile_products[0]

            if m2.HasSubstructMatch(patt2) == True:
                rxn1 = AllChem.ReactionFromSmarts('[CH1:1]#[CH1:2].[O:3]>>[CH:1](=[O:3])[CH1:2]', useSmiles=True)
                ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("O")))
                smile_products.append(Chem.MolToSmiles(ps1[0][0]))
                del(smile_products[0])
                smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def Lindlar_catalyst_reduction(smile):
        smile_products = [smile]
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile1 = smile_products[0]
        
        while FunctionalGroups.terminal_alkyne(smile1) == True or FunctionalGroups.internal_alkyne(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[H:3][H:4]>>[C:1](/[H:3])=[C:2](/[H:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][H]")))
            # smile_products.append(Chem.MolToSmiles(ps1[1][0]))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))

            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def lithium_ammonia_reduction(smile):
        smile_products = [smile]
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile1 = smile_products[0]
        
        while FunctionalGroups.terminal_alkyne(smile1) == True or FunctionalGroups.internal_alkyne(smile1) == True:
            rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C:2].[H:3][H:4]>>[C:1](/[H:3])=[C:2](\[H:4])', useSmiles=True)
            ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles("[H][H]")))
            # smile_products.append(Chem.MolToSmiles(ps1[1][0]))
            smile_products.append(Chem.MolToSmiles(ps1[0][0]))

            del(smile_products[0])
            smile1 = smile_products[0]

        return smile_products

    @staticmethod
    def acetylide_anion(smile):
        smile_products = [smile]
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        smile1 = smile_products[0]
        
        m1 = Chem.MolFromSmiles(smile1)
        patt1 = Chem.MolFromSmarts("CC#[CH]")

        while m1.HasSubstructMatch(patt1) == True:
            repl = Chem.MolFromSmarts('[C][C]#[C-]')
            patt = Chem.MolFromSmarts('[C][C]#[CH]')
            m = Chem.MolFromSmiles(smile1)
            rms = AllChem.ReplaceSubstructs(m,patt,repl)
            smile_products.append(Chem.MolToSmiles(rms[0]))
            del(smile_products[0])
            smile1 = smile_products[0]
            m1 = Chem.MolFromSmiles(smile1)

        return smile_products

    @staticmethod
    def alkylation_of_acetylide_anion(smile, finalSMILE):
        t = time.time()
        smile1 = smile
        finalSMILE = Chem.MolFromSmiles(finalSMILE)
        smile_products = []

        if FunctionalGroups.acetylide_anion(smile1) == True:

            # Checks if the unreduced alkyne can be found in the final product    
            repl1 = Chem.MolFromSmarts('[C][C]#[C]')
            patt1 = Chem.MolFromSmarts('[C][C]#[C-]')
            m1 = Chem.MolFromSmiles(smile1)
            rms1 = AllChem.ReplaceSubstructs(m1, patt1, repl1)    
            smile11 = Chem.MolFromSmarts(Chem.MolToSmiles(rms1[0]))

            # Checks if the alkyne has been reduced to an alkene
            repl2 = Chem.MolFromSmarts('[C][C]=[C]')
            patt2 = Chem.MolFromSmarts('[C][C]#[C-]')
            m2 = Chem.MolFromSmiles(smile1)
            rms2 = AllChem.ReplaceSubstructs(m2, patt2, repl2)    
            smile12 = Chem.MolFromSmarts(Chem.MolToSmiles(rms2[0]))

            # Checks if the alkyne has been reduced to an alkane
            repl3 = Chem.MolFromSmarts('[C][C]-[C]')
            patt3 = Chem.MolFromSmarts('[C][C]#[C-]')
            m3 = Chem.MolFromSmiles(smile1)
            rms3 = AllChem.ReplaceSubstructs(m3, patt3, repl3)    
            smile13 = Chem.MolFromSmarts(Chem.MolToSmiles(rms3[0]))

            if finalSMILE.HasSubstructMatch(smile11) == True:
                tmp = Chem.ReplaceCore(finalSMILE, Chem.MolFromSmarts(Chem.MolToSmarts(smile11)[::-1])) # This flips the string, since core replacement is sensitive to character order in string
                Chem.MolToSmiles(tmp)
                rs = Chem.GetMolFrags(tmp, asMols=True)
                tmp2 = Chem.ReplaceCore(finalSMILE, Chem.MolFromSmarts(Chem.MolToSmarts(smile11)))
                Chem.MolToSmiles(tmp2)
                rs = rs + Chem.GetMolFrags(tmp2, asMols=True)
                fragments = []

                for i in range(len(rs)):
                    fragment_i = Chem.MolToSmiles(rs[i])
                    for j in fragment_i:
                        if j.isdigit():
                            fragment_i = fragment_i.replace(j, "100")
                    if fragment_i not in fragments:
                        fragments.append(fragment_i)
                
                for j in range(len(fragments)):
                    m = Chem.MolFromSmiles(fragments[j])
                    patt = Chem.MolFromSmarts("[100*]C")
                    FG_found = m.HasSubstructMatch(patt)

                    if FG_found == True:
                        rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C-].[100*:3][C:4]>>[C:1]#[C][C:4].[100*:3]', useSmiles=True)
                        ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles(fragments[j])))
                        if Chem.MolToSmiles(ps1[0][0]) not in smile_products:
                            smile_products.append(Chem.MolToSmiles(ps1[0][0]))

            elif finalSMILE.HasSubstructMatch(smile12) == True:
                tmp = Chem.ReplaceCore(finalSMILE, Chem.MolFromSmarts(Chem.MolToSmarts(smile12)[::-1]))
                Chem.MolToSmiles(tmp)
                rs = Chem.GetMolFrags(tmp, asMols=True)
                tmp2 = Chem.ReplaceCore(finalSMILE, Chem.MolFromSmarts(Chem.MolToSmarts(smile12)))
                Chem.MolToSmiles(tmp2)
                rs = rs + Chem.GetMolFrags(tmp2, asMols=True)
                fragments = []

                for i in range(len(rs)):
                    fragment_i = Chem.MolToSmiles(rs[i])
                    for j in fragment_i:
                        if j.isdigit():
                            fragment_i = fragment_i.replace(j, "100")
                    if fragment_i not in fragments:
                        fragments.append(fragment_i)
                
                for j in range(len(fragments)):
                    m = Chem.MolFromSmiles(fragments[j])
                    patt = Chem.MolFromSmarts("[100*]C")
                    FG_found = m.HasSubstructMatch(patt)

                    if FG_found == True:
                        rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C-].[100*:3][C:4]>>[C:1]#[C][C:4].[100*:3]', useSmiles=True)
                        ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles(fragments[j])))
                        if Chem.MolToSmiles(ps1[0][0]) not in smile_products:
                            smile_products.append(Chem.MolToSmiles(ps1[0][0]))

            elif finalSMILE.HasSubstructMatch(smile13) == True:
                tmp = Chem.ReplaceCore(finalSMILE, Chem.MolFromSmarts(Chem.MolToSmarts(smile13)[::-1]))
                Chem.MolToSmiles(tmp)
                rs = Chem.GetMolFrags(tmp, asMols=True)
                tmp2 = Chem.ReplaceCore(finalSMILE, Chem.MolFromSmarts(Chem.MolToSmarts(smile13)))
                Chem.MolToSmiles(tmp2)
                rs = rs + Chem.GetMolFrags(tmp2, asMols=True)
                fragments = []

                for i in range(len(rs)):
                    fragment_i = Chem.MolToSmiles(rs[i])
                    for j in fragment_i:
                        if j.isdigit():
                            fragment_i = fragment_i.replace(j, "100")
                    if fragment_i not in fragments:
                        fragments.append(fragment_i)
                
                for j in range(len(fragments)):
                    m = Chem.MolFromSmiles(fragments[j])
                    patt = Chem.MolFromSmarts("[100*]C")
                    FG_found = m.HasSubstructMatch(patt)

                    if FG_found == True:
                        rxn1 = AllChem.ReactionFromSmarts('[C:1]#[C-].[100*:3][C:4]>>[C:1]#[C][C:4].[100*:3]', useSmiles=True)
                        ps1 = rxn1.RunReactants((Chem.MolFromSmiles(smile1),Chem.MolFromSmiles(fragments[j])))
                        if Chem.MolToSmiles(ps1[0][0]) not in smile_products:
                            smile_products.append(Chem.MolToSmiles(ps1[0][0]))
        print(f'{time.time() - t:.5f} sec')
        print(smile_products)
        print(fragments)

    @staticmethod
    def getAllReactions():
        method_list = [eval("AlkyneReactions." + method) for method in dir(AlkyneReactions) if method.startswith('_') is False and method.startswith("getAllReactions") is False]
        return method_list