import transformers
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor, BeitForImageClassification
from transformers import CLIPProcessor, CLIPModel
from  torch.cuda.amp import autocast
from PIL import Image
import requests
import glob
import os
from scipy import spatial
import pandas as pd
from tqdm import tqdm
import torch
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances

torch.cuda.empty_cache()


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_image(image_filename):
    # print(image_filename)
    image = Image.open(image_filename)
    img = image.copy()
    image.close()
    img = np.array(img)
    img = img[:, :, :3]
    return img


def load_images_folder(folder_path):
    images = []
    for img_filename in tqdm(os.listdir(folder_path)):
        if img_filename==".DS_Store":
            continue
        img = load_image(os.path.join(folder_path, img_filename))
        images.append(img)
    return images


def process_input(images_list, processor):
    # print("Number of images")
    # print(len(images_list))
    # inputs = processor(images=images_list, return_tensors="pt")
    #If CLIP model:
    inputs = processor(text=["a photo of a cat"], images=images_list, return_tensors="pt")
    return inputs


def get_vit_features(model, inputs):
    with torch.no_grad():
        with autocast():
            outputs = model(**inputs)
            # logits = outputs.logits
            #If CLIP is used:
            image_embedding = outputs.last_hidden_state[:, 0, :]
            print(image_embedding.shape)
    return image_embedding


def get_cosine_similarity_for_two_images(features_1, features_2):
    # cosine similarity
    cosine_similarity = 1 - spatial.distance.cosine(features_1, features_2)
    return cosine_similarity


def get_filenames(source_dir):
    l = os.listdir(source_dir)
    if ".DS_Store" in l:
        l.remove(".DS_Store")
    return l


def get_all_results(caps_filenames, outputs_caps, streams_filenames, outputs_streams):
    print("outputs_cap shape: ", len(outputs_caps))
    print("outputs_streams shape: ", len(outputs_streams))

    similarities = cosine_similarity(outputs_caps, outputs_streams)
    print(similarities.shape)

    results = {}
    for i, cap_filename in enumerate(caps_filenames):
        results[cap_filename] = {}
        for j, stream_filename in enumerate(streams_filenames):
            results[cap_filename][stream_filename] = similarities[i, j]

    return results


def run_model_on_batch(list_images, processor, model):
    outputs = []
    for img in tqdm(list_images):
        # print("Processing input...")
        input = process_input(img, processor)
        # print("Computing features...")
        torch.cuda.empty_cache()
        # output = get_vit_features(model, input.to(device, torch.float16))
        # if CLIP used:
        output = get_vit_features(model, input.to(device))
        outputs.append(output.cpu().detach().numpy())
    outputs = np.vstack(outputs)
    return outputs


def main():
    parser = argparse.ArgumentParser(description='Upload test args')
    parser.add_argument('-model_name', type=str, help="Name of model to use", default="google/vit-base-patch16-224")
    parser.add_argument('-caps_path', type=str, nargs='?', help='Path to captures images')
    parser.add_argument('-streams_path', type=str, nargs='?', help='Path to streams images')
    parser.add_argument('-results_path', type=str, nargs='?', help='Path to results')
    args = parser.parse_args()

    model_name = args.model_name
    caps_path = args.caps_path
    streams_path = args.streams_path
    path_to_results = args.results_path

    print("Loading model and processor...")
    # processor = ViTImageProcessor.from_pretrained(model_name)
    # processor = AutoImageProcessor.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    # model = ViTForImageClassification.from_pretrained(model_name)
    # model = BeitForImageClassification.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    model = model.to(device)


    print("Loading images...")
    caps_imgs = load_images_folder(caps_path)[:5]
    caps_filenames = get_filenames(caps_path)[:5]
    streams_imgs = load_images_folder(streams_path)[:5]
    streams_filenames = get_filenames(streams_path)[:5]


    print("Processing captures...")
    outputs_cap = run_model_on_batch(caps_imgs, processor, model)


    print("Processing streams...")
    outputs_stream = run_model_on_batch(streams_imgs, processor, model)

    # inputs_cap = process_input(caps_imgs, processor)
    # inputs_stream = process_input(streams_imgs, processor)

    # torch.cuda.empty_cache()

    # print("Computing features...")
    # outputs_cap = get_vit_features(model, inputs_cap.to(device, torch.float16))
    # outputs_stream = get_vit_features(model, inputs_stream.to(device, torch.float16))

    print("Getting all results...")
    all_res = get_all_results(caps_filenames, outputs_cap, streams_filenames,
                              outputs_stream)

    print("Saving results to :" , path_to_results)
    pd.DataFrame(all_res).to_csv(path_to_results)



if __name__ == '__main__':
    main()