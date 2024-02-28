import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors  import *
from rdkit.Chem import Draw
from PIL import Image
import io
import base64


def mol_to_image_base64(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            img = Draw.MolToImage(mol)
            output = io.BytesIO()
            img.save(output, format='PNG')
            return base64.b64encode(output.getvalue()).decode('utf-8')
        else:
            return None
    except Exception as e:
        st.error(f"Error generating image for SMILES: {smiles} | Error: {str(e)}")
        return None

# Function to compute Morgan fingerprints
def compute_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
        return np.array(fp)
    else:
        return None

# Load the model
model_path = 'Hydrogenation_random_forest_regressor_model.joblib'
loaded_model = joblib.load(model_path)

def SatMySmiles(molecule_smiles):   ## FIX THIS the triple bond should then change into a double bond

    new_smiles = ''
    nH2 = 0
    for i in molecule_smiles:
        if (i == '=' or i == '#'):
            nH2 = nH2 + 1
 
        elif (i == 'c' or i == 'n' or i == 'o' or i == 's'):
            i = i.upper()
            new_smiles = new_smiles+i  
        else:
            new_smiles = new_smiles+i
           
    return new_smiles

def percentWeightH2(smiles_dehydrogenated,smiles_hydrogenated):
    MW_hydrogenated = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles_hydrogenated))
    MW_dehydrogenated = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles_dehydrogenated))
    pH2 = ((MW_hydrogenated - MW_dehydrogenated)/(MW_hydrogenated)) * 100
    return pH2

# Streamlit UI
st.title('∆H Predictor for hydrogenation reactions')
st.write('This app predicts the delta H for hydrogenation reactions based on SMILES strings.')
#st.write('All units are given in kJ/mol H2')
st.markdown('All units are given in kJ/mol H<sub>2</sub>', unsafe_allow_html=True)
st.markdown('Random Forest Model, trained on G4MP2 data of 10k reactions from QM9-LOHC dataset (up to 9 heavy atoms)')
# User input for SMILES string
user_input_smiles = st.text_input("Enter SMILES string:", "c1cc(C)ccc1")

# Button to make predictions
if st.button('Predict ∆H'):
    # Convert the input SMILES string to Morgan fingerprint
    fp = compute_morgan_fingerprint(user_input_smiles)
    if fp is not None:
        saturated_smiles = SatMySmiles(user_input_smiles)
        pH2 = np.around((percentWeightH2(user_input_smiles,saturated_smiles)),2)
        st.write(f'%H2 by weight: {pH2}')
        if pH2 == 0:
           st.write("%wt H2 is zero, molecule is fully saturated!")
           exit()
        # Reshape for a single sample prediction
        fp = np.array([fp])
        # Predict delta_H value
        prediction = np.around(loaded_model.predict(fp),2)
        mol = Chem.MolFromSmiles(user_input_smiles)
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        st.write(f"Number of heavy atoms: {num_heavy_atoms}")
        st.write(f'Predicted Delta H: {prediction[0]} kJ/mol')
        saturated_smiles = SatMySmiles(user_input_smiles)
        st.write(f'Saturated SMILES: {saturated_smiles}')
        molecule1_img = mol_to_image_base64(user_input_smiles)
        molecule2_img = mol_to_image_base64(saturated_smiles)

### working ++
        col1, col2 = st.columns(2)
        with col1:
            if molecule1_img is not None:
                st.image(Image.open(io.BytesIO(base64.b64decode(molecule1_img))), caption="Unsaturated")
        with col2:
            if molecule2_img is not None:
                st.image(Image.open(io.BytesIO(base64.b64decode(molecule2_img))), caption="Saturated")
### working --

    else:
        st.write("Invalid SMILES string. Please enter a valid one.")
    

