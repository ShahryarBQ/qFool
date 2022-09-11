import os
import sys
from copy import deepcopy
import numpy as np
import eagerpy as ep
from PIL import Image
import matplotlib.pyplot as plt

import foolbox as fb
# from qfool import QFoolAttack
import torch
import torchvision.models as models
import torchvision.transforms as transforms


def main():
	# parse the input arguments
	attack_str = sys.argv[1]
	model_str = sys.argv[2]
	dataset_str = sys.argv[3]
	input_image_address_str = sys.argv[4]
	print(f"{model_str}\n{dataset_str}\n{attack_str}\n{input_image_address_str}")

	# use the arguments to determine the attack, the model, the dataset and the image
	attack = determine_attack(attack_str)
	model = determine_model(model_str)
	model.eval()
	dataset = determine_dataset(dataset_str)
	image = Image.open(input_image_address_str)

	# create the foolbox model
	mean, std, bounds, batchsize, epsilon = constants()
	preprocessing = dict(mean=mean, std=std, axis=-3)
	fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
	fmodel = fmodel.transform_bounds(bounds)

	attack_single_image(attack, fmodel, dataset, image, epsilon)
	# attack_multiple_images()


def attack_single_image(attack, fmodel, dataset, image, epsilon):
	# reformat the image according with its dataset and model, then query its label
	image_tensor = reformat_image(image, dataset, fmodel)
	image_dim = calculate_image_dim(image_tensor)
	label_orig = fmodel(image_tensor).argmax(axis=-1)

	# use the reformatted image to initiate the attack
	raw, clipped, is_adv = attack(fmodel, image_tensor, label_orig, epsilons=epsilon)
	# print(is_adv.item())
	pert = clipped[0] - image_tensor[0]
	clipped_image = transforms.ToPILImage()(reformat_tensor(clipped[0], fmodel, fmodel.bounds))
	pert_image = transforms.ToPILImage()(reformat_tensor(pert, fmodel, (-0.1,0.1)))

	# determine the output label
	label_adv = fmodel(clipped).argmax(axis=-1)
	labels = open(os.path.join(os.getcwd(), 'misc','synset_words.txt'), 'r').read().split('\n')
	str_label_orig = labels[label_orig.item()].split(',')[0].split()[1]
	str_label_adv = labels[label_adv.item()].split(',')[0].split()[1]

	# if the directory does not exist, create it
	savedir = os.path.join(os.getcwd(), "misc", "misc_temp")
	if not os.path.exists(savedir):
		os.makedirs(savedir)

	# save the images and print out the required data to console
	# the java code will use all of these
	clipped_image.save(os.path.join(savedir, "adv.jpg"))
	pert_image.save(os.path.join(savedir, "pert.jpg"))
	print(f"mse:{torch.linalg.norm(pert).item() / image_dim}")
	print(f"ori:{str_label_orig}")
	print(f"adv:{str_label_adv}")
	print("done")


def reformat_image(image, dataset, fmodel):
	x = deepcopy(image)
	if dataset == "imagenet":
		x = x.resize((224, 224))
	x = np.asarray(x, dtype=np.float32)
	if x.ndim == 2:
		x = x[..., np.newaxis]
	if fmodel.data_format == "channels_first":
		x = np.transpose(x, (2, 0, 1))
	if fmodel.bounds != (0, 255):
		x = x / 255 * (fmodel.bounds[1] - fmodel.bounds[0]) + fmodel.bounds[0]
	if hasattr(fmodel, "dummy") and fmodel.dummy is not None:  # type: ignore
		x = ep.from_numpy(fmodel.dummy, x).raw  # type: ignore
	x = x[None, ...]
	return x


def reformat_tensor(image, fmodel, bounds):
	x = ep.astensor(image).numpy()
	min_, max_ = bounds
	x = (x - min_) / (max_ - min_)
	return torch.tensor(x)


def constants():
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	bounds = (0, 1)
	batchsize = 16
	# epsilons = np.linspace(0.0, 0.005, num=20)
	epsilon = 0.01
	return mean, std, bounds, batchsize, epsilon


def calculate_image_dim(image):
	shape = image.numpy().shape
	dim = 1
	for s in shape:
		dim *= s
	return dim


def determine_attack(attack_str):
	attack = None
	if attack_str == "L2ContrastReductionAttack": 
		attack = fb.attacks.L2ContractReductionAttack()
	elif attack_str == "LinfContrastReductionAttack": 
		attack = fb.attacks.LinfContrastReductionAttack()
	elif attack_str == "VirtualAdversarialAttack": 
		attack = fb.attacks.VirtualAdversarialAttack()
	elif attack_str == "DecoupledDirectionAndNormAttack": 
		attack = fb.attacks.DDNAttack()
	elif attack_str == "L2ProjectedGradientDescentAttack": 
		attack = fb.attacks.L2ProjectedGradientDescentAttack()
	elif attack_str == "LinfProjectedGradientDescentAttack": 
		attack = fb.attacks.LinfProjectedGradientDescentAttack()
	elif attack_str == "L2BasicIterativeAttack": 
		attack = fb.attacks.L2BasicIterativeAttack()
	elif attack_str == "LinfBasicIterativeAttack": 
		attack = fb.attacks.LinfBasicIterativeAttack()
	elif attack_str == "L2FastGradientAttack": 
		attack = fb.attacks.L2FastGradientAttack()
	elif attack_str == "LinfFastGradientAttack": 
		attack = fb.attacks.LinfFastGradientAttack()
	elif attack_str == "AdditiveGaussianNoiseAttack": 
		attack = fb.attacks.L2AdditiveGaussianNoiseAttack()
	elif attack_str == "L2AdditiveUniformNoiseAttack": 
		attack = fb.attacks.L2AdditiveUniformNoiseAttack()
	elif attack_str == "LinfAdditiveUniformNoiseAttack": 
		attack = fb.attacks.LinfAdditiveUniformNoiseAttack()
	elif attack_str == "ClippingAwareAdditiveGaussianNoiseAttack": 
		attack = fb.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack()
	elif attack_str == "ClippingAwareAdditiveUniformNoiseAttack": 
		attack = fb.attacks.L2ClippingAwareAdditiveUniformNoiseAttack()
	elif attack_str == "RepeatedAdditiveGaussianNoiseAttack": 
		attack = fb.attacks.L2RepeatedAdditiveGaussianNoiseAttack()
	elif attack_str == "L2RepeatedAdditiveUniformNoiseAttack": 
		attack = fb.attacks.L2RepeatedAdditiveUniformNoiseAttack()
	elif attack_str == "LinfRepeatedAdditiveUniformNoiseAttack": 
		attack = fb.attacks.LinfRepeatedAdditiveUniformNoiseAttack()
	elif attack_str == "ClippingAwareRepeatedAdditiveGaussianNoiseAttack": 
		attack = fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack()
	elif attack_str == "ClippingAwareRepeatedAdditiveUniformNoiseAttack": 
		attack = fb.attacks.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack()
	elif attack_str == "InversionAttack": 
		attack = fb.attacks.InversionAttack()
	elif attack_str == "BinarySearchContrastReductionAttack": 
		attack = fb.attacks.BinarySearchContrastReductionAttack()
	elif attack_str == "LinearSearchContrastReductionAttack": 
		attack = fb.attacks.LinearSearchContrastReductionAttack()
	elif attack_str == "CarliniWagnerAttack": 
		attack = fb.attacks.L2CarliniWagnerAttack()
	elif attack_str == "NewtonFoolAttack": 
		attack = fb.attacks.NewtonFoolAttack()
	elif attack_str == "EADAttack": 
		attack = fb.attacks.EADAttack()
	elif attack_str == "GaussianBlurAttack": 
		attack = fb.attacks.GaussianBlurAttack()
	elif attack_str == "L2DeepFoolAttack": 
		attack = fb.attacks.L2DeepFoolAttack()
	elif attack_str == "LinfDeepFoolAttack": 
		attack = fb.attacks.LinfDeepFoolAttack()
	elif attack_str == "SaltAndPepperNoiseAttack": 
		attack = fb.attacks.SaltAndPepperNoiseAttack()
	elif attack_str == "LinearSearchBlendedUniformNoiseAttack": 
		attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack()
	elif attack_str == "BinarizationRefinementAttack": 
		attack = fb.attacks.BinarizationRefinementAttack()
	elif attack_str == "DatasetAttack": 
		attack = fb.attacks.DatasetAttack()
	elif attack_str == "BinaryAttack": 
		attack = fb.attacks.BoundaryAttack()
	elif attack_str == "L0BrendelBerthgeAttack": 
		attack = fb.attacks.L0BrendelBerthgeAttack()
	elif attack_str == "L1BrendelBerthgeAttack": 
		attack = fb.attacks.L1BrendelBerthgeAttack()
	elif attack_str == "L2BrendelBerthgeAttack": 
		attack = fb.attacks.L2BrendelBerthgeAttack()
	elif attack_str == "LinfBrendelBerthgeAttack": 
		attack = fb.attacks.LinfBrendelBerthgeAttack()
	# elif attack_str == "QFoolAttack":
	# 	attack = QFoolAttack()
	return attack


def determine_model(model_str):
	model = None
	if model_str == "AlexNet": 
		model = models.alexnet(pretrained=True)
	elif model_str == "VGG11": 
		model = models.vgg11(pretrained=True)
	elif model_str == "VGG13": 
		model = models.vgg13(pretrained=True)
	elif model_str == "VGG16": 
		model = models.vgg16(pretrained=True)
	elif model_str == "VGG19": 
		model = models.vgg19(pretrained=True)
	elif model_str == "VGG11bn": 
		model = models.vgg11_bn(pretrained=True)
	elif model_str == "VGG13bn": 
		model = models.vgg13_bn(pretrained=True)
	elif model_str == "VGG16bn": 
		model = models.vgg16_bn(pretrained=True)
	elif model_str == "VGG19bn": 
		model = models.vgg19_bn(pretrained=True)
	elif model_str == "ResNet18": 
		model = models.resnet18(pretrained=True)
	elif model_str == "ResNet34": 
		model = models.resnet34(pretrained=True)
	elif model_str == "ResNet50": 
		model = models.resnet50(pretrained=True)
	elif model_str == "ResNet101": 
		model = models.resnet101(pretrained=True)
	elif model_str == "ResNet152": 
		model = models.resnet152(pretrained=True)
	elif model_str == "SqueezeNet1_0": 
		model = models.squeezenet1_0(pretrained=True)
	elif model_str == "SqueezeNet1_1": 
		model = models.squeezenet1_1(pretrained=True)
	elif model_str == "DenseNet121": 
		model = models.densenet121(pretrained=True)
	elif model_str == "DenseNet161": 
		model = models.densenet161(pretrained=True)
	elif model_str == "DenseNet169": 
		model = models.densenet169(pretrained=True)
	elif model_str == "DenseNet201": 
		model = models.densenet201(pretrained=True)
	elif model_str == "Inceptionv3": 
		model = models.inception_v3(pretrained=True)
	elif model_str == "GoogLeNetInceptionv1": 
		model = models.googlenet(pretrained=True)
	elif model_str == "ShuffleNetV2x0_5": 
		model = models.shufflenet_v2_x0_5(pretrained=True)
	elif model_str == "ShuffleNetV2x1_0": 
		model = models.shufflenet_v2_x1_0(pretrained=True)
	elif model_str == "ShuffleNetV2x1_5": 
		model = models.shufflenet_v2_x1_5(pretrained=True)
	elif model_str == "ShuffleNetV2x2_0": 
		model = models.shufflenet_v2_x2_0(pretrained=True)
	elif model_str == "MobileNetV2": 
		model = models.mobilenet_v2(pretrained=True)
	elif model_str == "MobileNetV3Large": 
		model = models.mobilenet_v3_large(pretrained=True)
	elif model_str == "MobileNetV3Small": 
		model = models.mobilenet_v3_small(pretrained=True)
	elif model_str == "ResNeXt50_32x4d": 
		model = models.resnext50_32x4d(pretrained=True)
	elif model_str == "ResNeXt101_32x8d": 
		model = models.resnext101_32x8d(pretrained=True)
	elif model_str == "WideResNet50_2": 
		model = models.wide_resnet50_2(pretrained=True)
	elif model_str == "WideResNet101_2": 
		model = models.wide_resnet101_2(pretrained=True)
	elif model_str == "MNASNet0_5": 
		model = models.mnasnet0_5(pretrained=True)
	elif model_str == "MNASNet0_75": 
		model = models.mnasnet0_75(pretrained=True)
	elif model_str == "MNASNet1_0": 
		model = models.mnasnet1_0(pretrained=True)
	elif model_str == "MNASNet1_3": 
		model = models.mnasnet1_3(pretrained=True)
	return model


def determine_dataset(dataset_str):
	if dataset_str == "ImageNet":
		dataset = "imagenet"
	return dataset

	
if __name__ == "__main__":
	main()