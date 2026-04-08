import torch
import numpy as np
import joblib
from model_train_generate.model import ArmAutoencoder
import os


# load training model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ArmAutoencoder()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()  # generation mode, no learning

# scaler
scaler = joblib.load("scaler.pkl")

# movement generation
def generate_movement(num_sequences = 1):
    with torch.no_grad():

        # sample a random point in latent space
        latent = torch.randn(num_sequences, 8)  # latent_size from model file

        # decodes latent vector into a movement sequence
        dec_input = model.from_latent(latent)
        dec_input = dec_input.unsqueeze(1).repeat(1, 30, 1)  # 30 = WINDOW_SIZE
        output, _ = model.decoder(dec_input)

        return output.numpy()

# generates and converts into rela coordinates
generated = generate_movement(num_sequences = 10 )  # generate 10 movement sequences. Change for more or less
print(f"generated shape: {generated.shape}") 

# reshapes for scaler, then reshapes back 
num_seq, window, features = generated.shape
generated_flat = generated.reshape(-1, features)      
generated_real = scaler.inverse_transform(generated_flat) # convert 0-1 back to real coords
generated_real = generated_real.reshape(num_seq, window, features)

print("generation completed.")

# save sequences as generated_sequence + number of sequence
# row = frame
for i, seq in enumerate(generated_real):
    os.makedirs("generated", exist_ok=True)
    np.savetxt(f"generated/generated_sequence_{i+1}.csv", seq, delimiter = ",")