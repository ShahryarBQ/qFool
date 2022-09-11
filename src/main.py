import os
from PIL import Image
import torch
import torchvision.models as models
from qfool import qFool


class PaperPlots():
	def __init__(self):
		self.device = check_cuda()
		self.results = "../results"
		self.create_dir(os.path.join(os.getcwd(), self.results))

		print("***** ***** ***** ***** ***** ***** Figure 3 ***** ***** ***** ***** ***** *****")
		# self.figure3()
		print("***** ***** ***** ***** ***** ***** Figure 9 ***** ***** ***** ***** ***** *****")
		self.figure9()
		print("***** ***** ***** ***** ***** ***** Table 2 ***** ***** ***** ***** ***** *****")
		# self.table2()


	def figure3(self):
		images_dict = {"frog": Image.open(os.path.join("../data","qFool","frog.jpeg")),
					   "vase": Image.open(os.path.join("../data","qFool","vase.jpeg"))}
		nets_dict = {"vgg19": models.vgg19(pretrained=True),
					 "resnet50": models.resnet50(pretrained=True),
					 "inception_v3": models.inception_v3(pretrained=True)}
		method = 'non-targeted'
		max_queries = 20000
		# init_estimation = 2300	# for 3 iters
		init_estimation = 3400		# for 2 iters

		# self.attack_and_save(images_dict, nets_dict, method, max_queries, init_estimation, "figure3")
		self.attack_and_save(images_dict, nets_dict, method, max_queries, init_estimation, "figure3", subspacedim=75)


	def figure9(self):
		images_dict = {"cardoon": Image.open(os.path.join("../data","qFool","cardoon.jpeg"))}
		targets_dict = {"vase": Image.open(os.path.join("../data","qFool","vase.jpeg"))}
		nets_dict = {"vgg19": models.vgg19(pretrained=True)}
		method = 'targeted'
		max_queries = 50000
		# init_estimation = 5600	# for 3 iters
		init_estimation = 8400		# for 2 iters

		# self.attack_and_save(images_dict, nets_dict, method, max_queries, init_estimation, "figure9", targets_dict)
		self.attack_and_save(images_dict, nets_dict, method, max_queries, init_estimation, "figure9", targets_dict, subspacedim=75)


	def table2(self):
		images_dict = {"school_bus": Image.open(os.path.join("../data","qFool","school_bus.jpeg"))}
		targets_dict = {"french_loaf_1": Image.open(os.path.join("../data","qFool","french_loaf_1.jpeg")),
						"french_loaf_2": Image.open(os.path.join("../data","qFool","french_loaf_2.jpeg")),
						"french_loaf_3": Image.open(os.path.join("../data","qFool","french_loaf_3.jpeg")),
						"golden_retriever_1": Image.open(os.path.join("../data","qFool","golden_retriever_1.jpeg")),
						"golden_retriever_2": Image.open(os.path.join("../data","qFool","golden_retriever_2.jpeg")),
						"golden_retriever_3": Image.open(os.path.join("../data","qFool","golden_retriever_3.jpeg")),
						"sunflower_1": Image.open(os.path.join("../data","qFool","sunflower_1.jpeg")),
						"sunflower_2": Image.open(os.path.join("../data","qFool","sunflower_2.jpeg")),
						"sunflower_3": Image.open(os.path.join("../data","qFool","sunflower_3.jpeg"))}
		nets_dict = {"vgg19": models.vgg19(pretrained=True),
					 "resnet50": models.resnet50(pretrained=True),
					 "inception_v3": models.inception_v3(pretrained=True)}
		method = 'targeted'
		max_queries = 100000
		init_estimation = 12000		# for 3 iters
		# init_estimation = 17000		# for 2 iters

		# self.attack_and_save(images_dict, nets_dict, method, max_queries, init_estimation, "table2", targets_dict)
		self.attack_and_save(images_dict, nets_dict, method, max_queries, init_estimation, "table2", targets_dict, subspacedim=75)


	def attack_and_save(self, images_dict, nets_dict, method, max_queries,
		init_estimation, figurenum, targets_dict=None, subspacedim=None):

		self.create_dir(os.path.join(os.getcwd(), self.results, figurenum))

		qfool_dict = []
		for imagename, image in images_dict.items():
			for netname, net in nets_dict.items():
				if method == 'non-targeted':
					qfool_nontargeted = qFool(image, net, self.device, method=method,
						max_queries=max_queries//10, init_estimation=init_estimation//10, subspacedim=subspacedim)

					xadv, xpert, mse, str_label_orig, str_label_pert = qfool_nontargeted.run()
					self.save(figurenum, imagename, netname, xadv, xpert, mse, str_label_orig, str_label_pert)

				elif method == 'targeted':
					for targetname, target in targets_dict.items():
						qfool_targeted = qFool(image, net, self.device, target, method=method,
							max_queries=max_queries//10, init_estimation=init_estimation//10)

						xadv, xpert, mse, str_label_orig, str_label_pert = qfool_targeted.run()
						self.save(figurenum, imagename, netname, xadv, xpert, mse, str_label_orig, str_label_pert, targetname)


	def save(self, figurenum, imagename, netname, xadv, xpert, mse, str_label_orig, str_label_pert, targetname=None):
		if not targetname:
			path = os.path.join(os.getcwd(), self.results, figurenum, f"{imagename}_{netname}")
		elif targetname:
			path = os.path.join(os.getcwd(), self.results, figurenum, f"{imagename}_{netname}_{targetname}")
		xadv.save(f"{path}_adv.jpeg", format="jpeg")
		xpert.save(f"{path}_pert.jpeg", format="jpeg")
		filestr = f"mse = {torch.linalg.norm(mse)}\n"
		filestr += f"original label = {str_label_orig}\n"
		filestr += f"adversarial label = {str_label_pert}"
		with open(f"{path}.txt", 'w') as f:
			f.writelines(filestr)


	def create_dir(self, dirpath):
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)


def check_cuda(use_cuda=True):
			print(f"CUDA Available: {torch.cuda.is_available()}")
			if use_cuda and torch.cuda.is_available():
				device = torch.device("cuda")
				print("Using GPU ...")
			else:
				device = torch.device("cpu")
				print("Using CPU ...")
			return device


def main():
	paperplots = PaperPlots()


if __name__ == '__main__':
	main()