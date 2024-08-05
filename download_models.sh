#!/bin/bash

# Download parsing_atr.onnx
curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx" -o ckpt/humanparsing/parsing_atr.onnx

# Download parsing_lip.onnx
curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx" -o ckpt/humanparsing/parsing_lip.onnx

# Download body_pose_model.pth
curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth" -o ckpt/openpose/ckpts/body_pose_model.pth

# Download model_final_162be9.pkl
curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl?download=true" -o ckpt/densepose/model_final_162be9.pkl