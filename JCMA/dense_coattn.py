import math
import torch
import torch.nn as nn
import sys


class DenseCoAttn(nn.Module):
	def __init__(self, dim1, dim2, dim3, dropout):
		super(DenseCoAttn, self).__init__()
		self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(3)])

		dim=dim1 + dim2 + dim3
		self.key_linear = nn.Linear(dim, dim)


		self.query1_linear = nn.Linear(dim1, dim)
		self.query2_linear = nn.Linear(dim2, dim)
		self.query3_linear = nn.Linear(dim3, dim)


		self.value1_linear = nn.Linear(16165, 16165)
		self.value2_linear = nn.Linear(16165, 16165)
		self.value3_linear = nn.Linear(16165, 16165)

		self.relu = nn.ReLU()

	def forward(self, value1, value2, value3):

		joint = torch.cat((value1, value2, value3), dim=-1)
		key  = self.key_linear(joint)


		query1 = self.query1_linear(value1)
		query2 = self.query2_linear(value2)
		query3 = self.query3_linear(value3)


		value1 = self.value1_linear(value1.transpose(-2, -1))
		value2 = self.value2_linear(value2.transpose(-2, -1))
		value3 = self.value3_linear(value3.transpose(-2, -1))


		weighted1= self.qkv_attention(query1, key, value1, self.dropouts[0])
		weighted2= self.qkv_attention(query2, key, value2, self.dropouts[1])
		weighted3= self.qkv_attention(query3, key, value3, self.dropouts[2])

		return weighted1, weighted2, weighted3

	def qkv_attention(self, query, key, value, dropout=None):

		d_k = key.size(-1)
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  #C1
		attn_scores = torch.tanh(scores)

		if dropout:
			attn_scores = dropout(attn_scores)


		weighted = torch.matmul(value,attn_scores)
		weighted = torch.tanh(weighted)   #H1
		return self.relu(weighted)



