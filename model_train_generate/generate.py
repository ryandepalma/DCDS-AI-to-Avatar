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
def generate_movement(num_sequences=1):
    with torch.no_grad():
        generated = []

        for _ in range(num_sequences):
            # two random points in latent space — start and end pose
            latent_start = torch.randn(1, 8)
            latent_end   = torch.randn(1, 8)

            frames = []
            for t in range(150): # 150 = 7 seconds of movement
                # alpha moves from 0.0 to 1.0 across 30 frames
                alpha  = t / 149
                latent = (1 - alpha) * latent_start + alpha * latent_end

                # decode each frame from its own latent vector
                dec_input = model.from_latent(latent)
                dec_input = dec_input.unsqueeze(1)
                output, _ = model.decoder(dec_input)
                frames.append(output.squeeze().numpy())  # shape: (1, 18)

            generated.append(np.array(frames))  # shape: (30, 18)

        return np.array(generated)  # shape: (num_sequences, 30, 18)

# generates and converts into real coordinates
generated = generate_movement(num_sequences = 5)  # generate 10 movement sequences. Change for more or less
print(f"generated shape: {generated.shape}")

# reshapes for scaler, then reshapes back
num_seq, window, features = generated.shape
generated_flat = generated.reshape(-1, features)
generated_real = scaler.inverse_transform(generated_flat)  # convert 0-1 back to real coords
generated_real = generated_real.reshape(num_seq, window, features)

print("generation completed.")

# save sequences as generated_sequence + number of sequence
# row = frame
os.makedirs("generated", exist_ok=True)
for i, seq in enumerate(generated_real):
    np.savetxt(f"generated/generated_sequence_{i+1}.csv", seq, delimiter=",")