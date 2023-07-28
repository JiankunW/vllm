import torch 
from vllm.model_executor.input_metadata import InputMetadata

if True: 
    def merge_hidden_states(
        # self,
        cache_hs: torch.Tensor, 
        hs_l: torch.Tensor,
        hs_r: torch.Tensor,
        mt_l: InputMetadata,
        mt_r: InputMetadata,
        ori_tok_flag: torch.Tensor,
    ) -> torch.Tensor:
        

        # input_dict = {
        # "hs_l": hs_l, 
        # "hs_r": hs_r, 
        # "mt_l": mt_l, 
        # "mt_r": mt_r, 
        # "ori_tok_flag": ori_tok_flag,
        # "cache_hs": self.cache_hs
        # }
        # torch.save(input_dict, 'debug.pt')
        # exit(0)
        input_length_wo_pad_l = mt_l.tok_len_pred_flag.size(0)
        input_length_wo_pad_r = mt_r.tok_len_pred_flag.size(0)
        pos_tensor, neg_tensor = hs_l[:input_length_wo_pad_l], hs_r[:input_length_wo_pad_r]
        
        new_pos_tensor = torch.ones(pos_tensor.shape).cuda() # to(pos_tensor.device)
        new_cache_hs = torch.randn(self.cache_hs.shape).cuda() # to(self.cache_hs.device)

        new_ori_tok_flag = torch.ones(ori_tok_flag.shape)
        ori_tok_flag = new_ori_tok_flag.unsqueeze(-1).expand(new_cache_hs.shape[0], new_cache_hs.shape[1])
        torch.where(ori_tok_flag==1, new_pos_tensor, new_cache_hs)

        new_cache_hs[ori_tok_flag==1, :] = new_pos_tensor # pos_tensor
        
        if pos_tensor.shape[0] > 0: 
            self.cache_hs[ori_tok_flag==1, :] = pos_tensor
            # self.cache_hs[ori_tok_flag==1] = pos_tensor
        
        if neg_tensor.shape[0] > 0: 
            self.cache_hs[ori_tok_flag==0, :] = neg_tensor
        aligned_length = (-ori_tok_flag.size(0)) % 8
        self.cache_hs[len(ori_tok_flag):aligned_length] = 0
        # hs = merge_tensor_from_mask(pos_tensor, neg_tensor, ori_tok_flag)
        # hs = pad_2d_tensor_to_alignment(hs, 8, 1)
        return self.cache_hs[:aligned_length]
    
        input_length_wo_pad_l = mt_l.tok_len_pred_flag.size(0)
        input_length_wo_pad_r = mt_r.tok_len_pred_flag.size(0)
        pos_tensor, neg_tensor = hs_l[:input_length_wo_pad_l], hs_r[:input_length_wo_pad_r]
        
        # ori_tok_flag = new_ori_tok_flag.unsqueeze(-1).expand(new_cache_hs.shape[0], new_cache_hs.shape[1])
        # torch.where(ori_tok_flag==1, new_pos_tensor, new_cache_hs)
        # import pdb; pdb.set_trace() 
        # new_cache_hs[ori_tok_flag==1, :] = new_pos_tensor # pos_tensor
        
        if pos_tensor.shape[0] > 0: 
            cache_hs[ori_tok_flag==1, :] = pos_tensor
        
        if neg_tensor.shape[0] > 0: 
            cache_hs[ori_tok_flag==0, :] = neg_tensor

    input_dict = torch.load("debug.pt")
    # import pdb; pdb.set_trace() 
    merge_hidden_states(
        cache_hs=input_dict['cache_hs'], 
        hs_l=input_dict['hs_l'],
        hs_r=input_dict['hs_r'],
        mt_l=input_dict['mt_l'],
        mt_r=input_dict['mt_r'],
        ori_tok_flag=input_dict['ori_tok_flag']
    )