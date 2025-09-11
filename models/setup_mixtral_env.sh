#!/bin/bash

# === CONFIGURATION ===
HF_CACHE_DIR="/data4t/hf"
HF_TOKEN_ENV_FILE="/home/students/Leishmania/.env"  # ✅ Đã cập nhật

# === STEP 1: Create HF cache directories with proper permissions ===
echo "[1] Creating HuggingFace cache directories..."
sudo mkdir -p "$HF_CACHE_DIR/hub" "$HF_CACHE_DIR/transformers"
sudo chown -R $(whoami):$(whoami) "$HF_CACHE_DIR"

# === STEP 2: Export environment variables ===
echo "[2] Exporting HuggingFace environment variables..."
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"

# (Optional) Add to .bashrc for persistence
if ! grep -q "HF_HOME=" ~/.bashrc; then
    echo "export HF_HOME=\"$HF_CACHE_DIR\"" >> ~/.bashrc
    echo "export HF_HUB_CACHE=\"$HF_CACHE_DIR/hub\"" >> ~/.bashrc
    echo "export TRANSFORMERS_CACHE=\"$HF_CACHE_DIR/transformers\"" >> ~/.bashrc
fi

# === STEP 3: Increase file watcher limit to prevent VSCode warnings ===
echo "[3] Increasing file watcher limit..."
sudo bash -c 'echo fs.inotify.max_user_watches=524288 >> /etc/sysctl.conf'
sudo sysctl -p

# === STEP 4: Create .env if missing and remind to fill in HF_TOKEN ===
echo "[4] Checking HuggingFace token..."
if [ ! -f "$HF_TOKEN_ENV_FILE" ]; then
    echo "HF_TOKEN=your_huggingface_token_here" > "$HF_TOKEN_ENV_FILE"
    echo ">> 🔑 Please update your HuggingFace token in $HF_TOKEN_ENV_FILE"
else
    echo "✅ Found existing .env file at: $HF_TOKEN_ENV_FILE"
fi

# === STEP 5: Install tmux ===
echo "[5] Ensuring tmux is installed..."
sudo apt update && sudo apt install tmux -y

# === STEP 6: Print next steps ===
echo -e "\n✅ Done! Now you can run:"
echo "   tmux new -s mixtral_download"
echo "   # and inside tmux run your python download script or notebook"

# === Optional: Reload terminal env ===
echo -e "\n💡 Restart your shell or run: source ~/.bashrc"
