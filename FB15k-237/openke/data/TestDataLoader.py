# coding:utf-8
import os
import pdb
import ctypes
import torch
import numpy as np
from openke.module.datasets.elph import HashDataset
from openke.native import load_base_library

class TestDataSampler(object):

	def __init__(self, data_total, data_sampler):
		self.data_total = data_total
		self.data_sampler = data_sampler
		self.total = 0

	def __iter__(self):
		return self

	def __next__(self):
		self.total += 1
		if self.total > self.data_total:
			raise StopIteration()
		return self.data_sampler()

	def __len__(self):
		return self.data_total

class TestDataLoader(object):

	def __init__(self, in_path = "./", sampling_mode = 'link', type_constrain = True, device='cuda', max_hash_hops = 2, feature_mode='triple', hll_p=8,  minhash_num_perm=128, split='test'):
		self.lib = load_base_library()
		self.lib.setInPath.argtypes = [ctypes.c_char_p]
		self.lib.setTestPath.argtypes = [ctypes.c_char_p]
		self.lib.setValidPath.argtypes = [ctypes.c_char_p]
		self.max_hash_hops = max_hash_hops
		if feature_mode not in ('triple', 'pairwise', 'none'):
			raise ValueError(
				"feature_mode must be 'triple', 'pairwise', or 'none'"
			)
		self.feature_mode = feature_mode
		"""for link prediction"""
		self.lib.getHeadBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		self.lib.getTailBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		self.lib.getTestEdges.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		"""for triple classification"""
		self.lib.getTestBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		"""set essential parameters"""
		self.in_path = in_path
		self.sampling_mode = sampling_mode
		self.type_constrain = type_constrain
		self.device = device
		self.hll_p = hll_p
		self.minhash_num_perm = minhash_num_perm
		if split not in ('test', 'valid'):
			raise ValueError("split must be 'test' or 'valid'")
		self.split = split
		self.read()

	def read(self):
		self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
		# Base.so evaluates the file registered as test_file. Swapping the two
		# held-out files gives validation the exact same candidate generation,
		# filtering, tie handling, and metric implementation as final testing.
		if self.split == 'valid':
			test_path = os.path.join(self.in_path, 'valid2id.txt')
			valid_path = os.path.join(self.in_path, 'test2id.txt')
		else:
			test_path = os.path.join(self.in_path, 'test2id.txt')
			valid_path = os.path.join(self.in_path, 'valid2id.txt')
		self.lib.setTestPath(ctypes.create_string_buffer(test_path.encode()))
		self.lib.setValidPath(ctypes.create_string_buffer(valid_path.encode()))
		self.lib.randReset()
		self.lib.importTestFiles()

		if self.type_constrain:
			type_path = os.path.join(self.in_path, 'type_constrain.txt')
			if os.path.isfile(type_path):
				self.lib.importTypeFiles()
			else:
				# Several standard OpenKE datasets (including FB15k-237) do
				# not ship type constraints.  Calling Base.so here can segfault
				# while opening the missing native-side file, so use the same
				# unconstrained filtered ranking as Tester.run_link_prediction.
				self.type_constrain = False
				print("Type constraint file not found; using unconstrained ranking")

		self.relTotal = self.lib.getRelationTotal()
		self.entTotal = self.lib.getEntityTotal()
		self.testTotal = self.lib.getTestTotal()
		self.trainTotal = self.lib.getTrainTotal()
		self.tripleTotal = self.lib.getTripleTotal()

		self.test_h = np.zeros(self.entTotal, dtype=np.int64)
		self.test_t = np.zeros(self.entTotal, dtype=np.int64)
		self.test_r = np.zeros(self.entTotal, dtype=np.int64)
		self.test_h_addr = self.test_h.__array_interface__["data"][0]
		self.test_t_addr = self.test_t.__array_interface__["data"][0]
		self.test_r_addr = self.test_r.__array_interface__["data"][0]

		self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_h_addr = self.test_pos_h.__array_interface__["data"][0]
		self.test_pos_t_addr = self.test_pos_t.__array_interface__["data"][0]
		self.test_pos_r_addr = self.test_pos_r.__array_interface__["data"][0]
		self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_h_addr = self.test_neg_h.__array_interface__["data"][0]
		self.test_neg_t_addr = self.test_neg_t.__array_interface__["data"][0]
		self.test_neg_r_addr = self.test_neg_r.__array_interface__["data"][0]
		self.triples = self.get_triples()
		self.hash_dataset = None
		if self.feature_mode != 'none':
			self.hash_dataset = HashDataset(
				self.triples,
				self.entTotal + self.relTotal,
				max_hash_hops=self.max_hash_hops,
				hll_p=self.hll_p,
				minhash_num_perm=self.minhash_num_perm,
			)
		#pdb.set_trace()

	def get_triples(self):
		full_list_h = np.zeros(self.trainTotal * 3, dtype=np.int64)
		full_list_t = np.zeros(self.trainTotal * 3, dtype=np.int64)
		full_list_h_addr = full_list_h.__array_interface__["data"][0]
		full_list_t_addr = full_list_t.__array_interface__["data"][0]
		self.lib.getEdgeIndices.argtypes = [
			ctypes.c_long,
			ctypes.c_long
		]
		self.lib.getEdgeIndices(
			full_list_h_addr,
			full_list_t_addr
		)
		#pdb.set_trace()
		return torch.from_numpy(np.vstack([full_list_h, full_list_t])).to(self.device)

	def sampling_lp(self):
		res = []
		self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
		links = torch.from_numpy(np.vstack([self.test_h, self.test_t]).T).to(self.device)
		relationships = torch.from_numpy(np.add(self.test_r, self.entTotal)).to(self.device)
		subgraph_feature = self._extract_features(links, relationships)
		res.append({
			"batch_h": self.test_h.copy(),
			"batch_t": self.test_t[:1].copy(),
			"batch_r": self.test_r[:1].copy(),
			"subgraph_feature": subgraph_feature,
			"mode": "head_batch"
		})
		self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
		links = torch.from_numpy(np.vstack([self.test_h, self.test_t]).T).to(self.device)
		relationships = torch.from_numpy(np.add(self.test_r, self.entTotal)).to(self.device)
		subgraph_feature = self._extract_features(links, relationships)
		res.append({
			"batch_h": self.test_h[:1],
			"batch_t": self.test_t,
			"batch_r": self.test_r[:1],
			"subgraph_feature": subgraph_feature,
			"mode": "tail_batch"
		})
		return res

	def _extract_features(self, links, relationships):
		if self.feature_mode == 'none':
			return torch.empty(
				(len(links), 0),
				dtype=torch.float32,
				device=links.device,
			)
		hashes = self.hash_dataset.get_hashes()
		cards = self.hash_dataset.get_cards()
		if self.feature_mode == 'pairwise':
			return self.hash_dataset.elph_hashes.get_pairwise_features(
				links, hashes, cards
			)
		return self.hash_dataset.elph_hashes.get_subgraph_features(
			links, relationships, hashes, cards
		)

	def sampling_tc(self):
		self.lib.getTestBatch(
			self.test_pos_h_addr,
			self.test_pos_t_addr,
			self.test_pos_r_addr,
			self.test_neg_h_addr,
			self.test_neg_t_addr,
			self.test_neg_r_addr,
		)
		return [
			{
				'batch_h': self.test_pos_h,
				'batch_t': self.test_pos_t,
				'batch_r': self.test_pos_r ,
				"mode": "normal"
			},
			{
				'batch_h': self.test_neg_h,
				'batch_t': self.test_neg_t,
				'batch_r': self.test_neg_r,
				"mode": "normal"
			}
		]

	"""interfaces to get essential parameters"""

	def get_ent_tot(self):
		return self.entTotal

	def get_rel_tot(self):
		return self.relTotal

	def get_triple_tot(self):
		return self.testTotal

	def set_sampling_mode(self, sampling_mode):
		self.sampling_mode = sampling_mode

	def __len__(self):
		return self.testTotal

	def __iter__(self):
		if self.sampling_mode == "link":
			self.lib.initTest()
			return TestDataSampler(self.testTotal, self.sampling_lp)
		else:
			self.lib.initTest()
			return TestDataSampler(1, self.sampling_tc)
