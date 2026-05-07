# HTCLN
Hadamard Transform cov-Lstm net. 结合starnet和TCLN网络的新网络，用于MTS多元时间序列任务预测，预测值是具体数值。\n
若是需要使用starhead则直接导入下载starnet，使用“from starnet import StarHead”这段代码，其中使用时直接使用StarHead(in_dim=fuse_in, hidden_dim=int(star_hidden), out_dim=1)函数即可
