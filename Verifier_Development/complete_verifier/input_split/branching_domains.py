#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from typing import Union
import torch
from torch import Tensor
from typing import Union, Tuple
from input_split.split import get_split_depth
from tensor_storage import get_tensor_storage

class InputDomainList:
    """Abstract class that maintains a list of domains for input split."""

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        # get lb, dm_l, dm_u, cs, threshold for idx; for convenience, alpha and split_idx
        # are not returned for now
        raise NotImplementedError

    def add(self, lb, dm_l, dm_u, alpha, cs, threshold=0, split_idx=None,
            remaining_index=None):
        raise NotImplementedError

    def pick_out_batch(self, batch, device="cuda"):
        raise NotImplementedError

    def get_topk_indices(self, k=1, largest=False):
        # get the topk indices, by default worst k
        raise NotImplementedError


class UnsortedInputDomainList(InputDomainList):
    """Unsorted domain list for input split."""

    def __init__(self, storage_depth, use_alpha=False,
                 sort_index=None, sort_descending=True, use_split_idx=True, 
                 store_upper_bound=False, store_adv_example=False, store_inlist_round=True,
                 extra_item_list=[]):
        super(UnsortedInputDomainList, self).__init__()
        item_list = ["lb", "dm_l", "dm_u", "cs", "threshold"]
        item_list += extra_item_list
        if use_split_idx:
            item_list.append("split_idx")
        if store_upper_bound:
            item_list.append("ub")
        if store_adv_example:
            item_list.append("adv_example")
        if store_inlist_round:
            item_list.append("inlist_round")
        self.extra_item_list = extra_item_list
        self.item_list = item_list
        for item in item_list:
            setattr(self, item, item_storage())
        self.alpha = {}
        self.use_alpha = use_alpha
        self.sort_index = sort_index
        self.storage_depth = storage_depth
        self.store_upper_bound = store_upper_bound
        self.store_adv_example = store_adv_example
        self.store_inlist_round = store_inlist_round

        self.global_x_L = None
        self.global_x_U = None
        
        self.sort_descending = sort_descending
        self.volume = self.all_volume = None
        self.use_split_idx = use_split_idx

    def __len__(self):
        return getattr(self, self.item_list[0]).__len__()

    def __getitem__(self, idx):
        ret = []
        extra_item_dict = {}
        for item in self.item_list:
            if item in self.extra_item_list:
                extra_item_dict[item] = getattr(self, item).__getitem__(idx)
            ret.append(getattr(self, item).__getitem__(idx))
        ret.append(extra_item_dict)
        return ret

    def filter_verified_domains(
            self,
            batch: int,
            lb: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            alpha: Union[dict, list, None],
            cs: Tensor,
            threshold: Union[int, Tensor] = 0,
            lA: Union[Tensor, None] = None,
            lbias: Union[Tensor, None] = None,
            check_thresholds=True,
            check_bounds=True
    ) -> Tuple[int, Tensor, Tensor, Tensor, Union[dict, list, None],
    Tensor, Tensor, Union[Tensor, None], Union[Tensor, None]]:
        """
        Filters out the domains that are verified and only returns unverified domains
        @param batch:                                   Batch size of domains
        @param lb: (batch, spec_dim)                    Domain lower bound output
        @param dm_l: (batch, dim_in)                    Input domain lower input bound
        @param dm_u: (batch, dim_in)                    Input domain upper input bound
        @param alpha:                                   CROWN alpha parameters for domains
        @param cs: (batch, spec_dim, lA_rows)           specification matrix
        @param threshold: (batch, spec_dim)             Threshold to verify specification with
        @param lA: (batch, lA_rows or spec_dim, dim_in) CROWN lA coefficient matrix
        @param lbias: (batch, spec_dim)                 CROWN lbias coefficient matrix
        @param check_thresholds:                        If true, filters out domains that have been verified
                                                        by lb > threshold
        @param check_bounds:                            If true, filters out domains that have been verified
                                                        by dm_l < dm_u
        @param double_alphas:                           If true, the alphas are repeated along the batch dimension by
                                                        split_partitions ** split_depth
        @param split_partitions:                        The number of partitions that domains are split into for BaB
        @return:
        """
        remaining_index = self.get_remaining_index(
            batch, lb, threshold, dm_l, dm_u, check_thresholds, check_bounds
        )
        lb_filt = lb[remaining_index]
        dm_l_filt = dm_l[remaining_index]
        dm_u_filt = dm_u[remaining_index]
        cs_filt = cs[remaining_index]
        batch_filt = len(dm_l_filt)
        alpha_filt = []
        if self.use_alpha and batch_filt > 0:
            alpha_filt = self.filter_alpha(alpha, remaining_index)
        threshold_filt = threshold[remaining_index]
        lA_filt = lA[remaining_index] if lA is not None else None
        lbias_filt = lbias[remaining_index] if lbias is not None else None

        return batch_filt, lb_filt, dm_l_filt, dm_u_filt, alpha_filt, cs_filt, threshold_filt, lA_filt, lbias_filt

    def filter_alpha(
            self,
            alpha: dict,
            remaining_index: Tensor
    ) -> dict:
        """
        Filters alphas w.r.t. remaining_index
        @param alpha:               Dictionary of alpha parameters
        @param remaining_index:     Batch indices to retain, typically the unverified indices
        @return:                    Filtered alpha dictionary
        """
        is_tensor = isinstance(remaining_index, Tensor)
        with torch.no_grad():
            alpha_filt = {}
            on_device = False # ensure remaining_index is on the correct device
            for key0 in alpha.keys():
                alpha_filt[key0] = {}
                for key1 in alpha[key0].keys():
                    if not on_device and is_tensor:
                        remaining_index = remaining_index.to(device=alpha[key0][key1].device)
                        on_device=True
                    # alpha[key0][key1] has shape (dim_in, spec_dim, batches, unstable size)
                    alpha_filt[key0][key1] = alpha[key0][key1][:, :, remaining_index]

        return alpha_filt

    def get_remaining_index(
            self,
            batch: int,
            lb: Tensor,
            threshold: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            check_thresholds=True,
            check_bounds=True
    ) -> Union[Tensor, Tuple]:
        """
        Gets the indices of the batch instances that are not verified. Verification conditions are specified by
        the check_thresholds and check_bounds flags. If both are None, all indicies are returned.
        @param batch:                       Batch size of domains
        @param lb: (batch, spec_dim)        Domain lower bound output
        @param threshold: (batch, spec_dim) Threshold to verify specification with
        @param dm_l: (batch, dim_in)        Input domain lower input bound
        @param dm_u: (batch, dim_in)        Input domain upper input bound
        @param check_thresholds:            If true, filters out domains that have been verified by lb > threshold
        @param check_bounds:                If true, filters out domains that have been verified by dm_l < dm_u
        @return:                            The indices of the batch instances that are left unverified
        """

        if check_thresholds and check_bounds:
            return torch.where(
                torch.logical_and(
                    (lb <= threshold).all(1),
                    (dm_l.view(batch, -1) <= dm_u.view(batch, -1)).all(1)
                )
            )[0]
        elif check_thresholds:
            return torch.where(
                    (lb <= threshold).all(1)
            )[0]
        elif check_bounds:
            return torch.where(
                    (dm_l.view(batch, -1) <= dm_u.view(batch, -1)).all(1)
            )[0]
        else:
            return slice(None)

    def add(
            self,
            lb: Tensor,
            dm_l: Tensor,
            dm_u: Tensor,
            alpha: Tensor,
            cs: Tensor,
            threshold: Union[int, Tensor] = 0,
            split_idx: Union[Tensor, None] = None,
            remaining_index: Union[Tensor, None] = None,
            check_thresholds=True,
            check_bounds=True,
            ub=None, 
            adv_example=None,
            extra_item_dict={}
    ) -> None:
        """
        Takes verified and unverified subdomains and only adds the unverified subdomains
        @param lb: Shape (batch, input_dim)                 Lower bound on domain outputs
        @param dm_l: Shape (batch, num_spec)                Lower bound on domain inputs
        @param dm_u: Shape (batch, num_spec)                Upper bound on domain inputs
        @param alpha:                                       alpha parameters
        @param cs: Shape (batch, num_spec, lA rows)         The C transformation matrix
        @param threshold: Shape (batch, num_spec)           The specification thresholds
        @param split_idx: Shape (batch, num of splits)      Specifies along which dimensions to split
        @param remaining_index:                             If not None, user is specifying which domains are unverified
        @return:                                            None
        """
        # check shape correctness
        batch = len(lb)
        if batch == 0:
            return
        if self.use_split_idx:
            assert split_idx is not None, "Cannot accept split_idx"
            assert len(split_idx) == batch
            assert split_idx.shape[1] == self.storage_depth
        else:
            assert split_idx is None, "Expected to receive split_idx"
        if self.store_upper_bound:
            assert ub is not None
        if self.store_adv_example:
            assert adv_example is not None
        if type(threshold) == int:
            threshold = torch.zeros(batch, 2)
        assert len(dm_l) == len(dm_u) == len(cs) == len(threshold) == batch
        if self.use_alpha:
            if alpha is None:
                raise ValueError("alpha should not be None in alpha-crown.")
        # initialize attributes using input shapes
        if self.use_alpha and not self.alpha:
            if type(alpha) == list:
                assert len(alpha) > 0
                for key0 in alpha[0].keys():
                    self.alpha[key0] = {}
                    for key1 in alpha[0][key0].keys():
                        self.alpha[key0][key1] = get_tensor_storage(
                            alpha[0][key0][key1].shape, concat_dim=2
                        )
            else:
                for key0 in alpha.keys():
                    self.alpha[key0] = {}
                    for key1 in alpha[key0].keys():
                        self.alpha[key0][key1] = get_tensor_storage(
                            alpha[key0][key1].shape, concat_dim=2
                        )
        # compute unverified indices
        if remaining_index is None:
            remaining_index = self.get_remaining_index(
                batch, lb, threshold, dm_l, dm_u, check_thresholds, check_bounds
            )
            if isinstance(remaining_index, Tensor):
                remaining_index = remaining_index.detach().cpu()
        # append the tensors
        if self.store_inlist_round:
            locals()["inlist_round"] = torch.zeros([batch, 1], dtype=torch.get_default_dtype(), device=lb.device)
        for item in self.extra_item_list:
            locals()[item] = extra_item_dict[item]
        for item in self.item_list:
            getattr(self, item).append(locals()[item][remaining_index])

        dm_l = dm_l[remaining_index]
        dm_u = dm_u[remaining_index]
        self._add_volume(dm_l, dm_u)

        if self.use_alpha:
            if type(alpha) == list:
                for i in remaining_index:
                    for key0 in alpha[0].keys():
                        for key1 in alpha[0][key0].keys():
                            self.alpha[key0][key1].append(
                                alpha[i][key0][key1]
                                .type(self.alpha[key0][key1].dtype)
                                .to(self.alpha[key0][key1].device)
                            )
            else:
                for key0 in alpha.keys():
                    for key1 in alpha[key0].keys():
                        self.alpha[key0][key1].append(
                            alpha[key0][key1][:, :, remaining_index]
                            .type(self.alpha[key0][key1].dtype)
                            .to(self.alpha[key0][key1].device)
                        )
        self.show_stat()

    # ANONYMOUS: pick out batch with softmax sampling
    def pick_out_batch_softmax(self, batch, temperature, device="cuda", use_upper_bound=False, use_adv_example=False, use_inlist_round=False):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        use_upper_bound = use_upper_bound and self.store_upper_bound
        batch = min(self.__len__(), batch)
        # assert batch > 0, "List of InputDomain is empty; pop failed."
        if batch == 0:
            return None
        print(f'picking out batch {batch} from {self.__len__()} domains')
        # ANONYMOUS: softmax sampling works better than picking adversarial examples
        # if use adv exmaple, pick up top cases with lowest upper bound
        # else, pick up cases with larger ub-lb if use_upper_bound and smaller lb if not use_upper_bound by softmax sampling
        if use_inlist_round:
            storage = getattr(self, "inlist_round").get_all()
            scores = (storage-storage.min())/(storage.max()-storage.min()).clamp(min=1e-6)
            softmax_scores = torch.softmax(scores / temperature, dim=0).flatten()
            sampled_indices = torch.multinomial(softmax_scores, batch, replacement=False)
            print("softmax score", softmax_scores[sampled_indices].min(), softmax_scores[sampled_indices].max())
        elif use_adv_example and self.store_adv_example:
            # Filter out indices that are already adversarial, i.e., item!=inf
            storage = getattr(self, "adv_example").get_all()
            mask = ~torch.isinf(storage)
            num_dims = storage.ndim
            for i in range(num_dims - 1):
                mask = mask.any(dim=-1)
            adv_indices = torch.nonzero(mask).squeeze(-1)
            num_adv_cases = len(adv_indices)
            print(f'num_adv_cases: {num_adv_cases}')
            if num_adv_cases >= batch:
                # # If there are enough or more adversarial examples, sample based on upper bound
                _, sorted_indices = torch.sort(getattr(self, "ub").get_all()[adv_indices], dim=0)
                sampled_indices = adv_indices[sorted_indices.squeeze(-1)][:batch]
            else:
                # If there are not enough adversarial examples, take all and then sample additional cases
                remaining_batch = batch - num_adv_cases
                storage = getattr(self, "ub").get_all() - getattr(self, "lb").get_all() if use_upper_bound else -getattr(self, "lb").get_all()
                non_adv_indices = torch.tensor([i for i in range(self.__len__()) if i not in adv_indices.tolist()], dtype=torch.long)
                scores = storage[non_adv_indices]
                scores = (scores-scores.min())/(scores.max()-scores.min()).clamp(min=1e-6)

                softmax_scores = torch.softmax(scores / temperature, dim=0).flatten()
                additional_sampled_indices = torch.multinomial(softmax_scores, remaining_batch, replacement=False)
                additional_sampled_indices = non_adv_indices[additional_sampled_indices]
                sampled_indices = torch.cat([adv_indices, additional_sampled_indices])
                print("softmax score", softmax_scores[sampled_indices].min(), softmax_scores[sampled_indices].max())
        else:
            # Original sampling process
            if use_upper_bound:
                storage = -getattr(self, "ub").get_all()
            else: 
                storage = -getattr(self, "lb").get_all()
            # ANONYMOUS: if we skip bounding before, lb will be all zeros
            # only use upper bound for sampling, smaller upper bound is better
            if torch.all(getattr(self, "lb").get_all() == 0):
                storage = -getattr(self, "ub").get_all()
            scores = (storage-storage.min())/(storage.max()-storage.min()).clamp(min=1e-6)

            softmax_scores = torch.softmax(scores / temperature, dim=0).flatten()
            sampled_indices = torch.multinomial(softmax_scores, batch, replacement=False)
            print("softmax score", softmax_scores[sampled_indices].min(), softmax_scores[sampled_indices].max())
        # sampled_indices, _ = sampled_indices.sort()
        remaining_indices = torch.ones(self.__len__(), dtype=torch.bool)
        remaining_indices[sampled_indices] = False

        for item in self.item_list:
            locals()[item] = getattr(self, item).remove(sampled_indices, device)
        alpha, val = [], []
        if self.use_alpha:
            for key0, val0 in self.alpha.items():
                for key1, val1 in val0.items():
                    val.append(val1.pop(batch))
            for i in range(batch):
                val_idx, item = 0, {}
                for key0, val0 in self.alpha.items():
                    item[key0] = {}
                    for key1 in val0.keys():
                        item[key0][key1] = val[val_idx][:, :, i: i + 1].to(
                            device=device, non_blocking=True
                        )
                        val_idx += 1
                alpha.append(item)
        # update num_used and the storage
        num_used = self.__len__()
        new_num_used = num_used - batch
        if self.use_alpha:
            for key0 in self.alpha.keys():
                for key1 in self.alpha[key0].keys():
                    self.alpha[key0][key1]._storage[:new_num_used] = \
                            self.alpha[key0][key1]._storage[:num_used][remaining_indices]
        if self.use_alpha:
            for key0 in self.alpha.keys():
                for key1 in self.alpha[key0].keys():
                    self.alpha[key0][key1].num_used = new_num_used
        ret_dict = {"alpha": alpha}
        extra_item_dict = {}
        for item in self.item_list:
            if item in self.extra_item_list:
                extra_item_dict[item] = locals()[item]
            else:
                ret_dict[item] = locals()[item]
        ret_dict["extra_item_dict"] = extra_item_dict
        return ret_dict

    def pick_out_batch(self, batch, device="cuda"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch = min(len(self), batch)
        assert batch > 0, "List of InputDomain is empty; pop failed."
        for item in self.item_list:
            locals()[item] = getattr(self, item).remove(batch, device)
        alpha, val = [], []
        if self.use_alpha:
            for key0, val0 in self.alpha.items():
                for key1, val1 in val0.items():
                    val.append(val1.pop(batch))
            for i in range(batch):
                val_idx, item = 0, {}
                for key0, val0 in self.alpha.items():
                    item[key0] = {}
                    for key1 in val0.keys():
                        item[key0][key1] = val[val_idx][:, :, i: i + 1].to(
                            device=device, non_blocking=True
                        )
                        val_idx += 1
                alpha.append(item)
        
        self._add_volume(locals()['dm_l'], locals()['dm_u'], sign=-1)
        ret_dict = {"alpha": alpha}
        for item in self.item_list:
            ret_dict[item] = locals()[item]
        return ret_dict

    def _add_volume(self, dm_l, dm_u, sign=1):
        volume = torch.prod(dm_u - dm_l, dim=-1).sum().item()
        if self.all_volume is None:
            self.all_volume = volume
            self.volume = 0
        self.volume = self.volume + sign * volume

    def get_progess(self):
        if self.all_volume is None or self.all_volume == 0:
            return 0.
        else:
            return 1 - self.volume / self.all_volume

    def _get_sort_margin(self, margin):
        if self.sort_index is not None:
            return margin[..., self.sort_index]
        else:
            return margin.max(dim=1).values

    def get_topk_indices(self, k=1, largest=False, use_upper_bound=False):
        assert k <= self.__len__(), print("Asked indices more than domain length.")
        use_upper_bound = use_upper_bound and self.store_upper_bound
        if use_upper_bound:
            ub = getattr(self, "ub").get_all()
            indices = ub.max(dim=1).values.topk(k, largest=largest).indices
        else:
            lb = getattr(self, "lb").get_all()
            threshold = getattr(self, "threshold").get_all()
            indices = self._get_sort_margin(lb - threshold).topk(k, largest=largest).indices
        return indices

    def sort(self, use_upper_bound=False):
        lb = getattr(self, "lb").get_all()
        threshold = getattr(self, "threshold").get_all()
        use_upper_bound = use_upper_bound and self.store_upper_bound
        if use_upper_bound:
            ub = getattr(self, "ub").get_all()
            indices = ub.max(dim=1).values.argsort(descending=True)
        else:
            indices = self._get_sort_margin(lb - threshold).argsort(
                descending=self.sort_descending)
        # sort the storage
        for item in self.item_list:
            getattr(self, item).update(indices)
        if self.use_alpha:
            for val0 in self.alpha.values():
                for val1 in val0.values():
                    val1._storage[
                    :, :, :val1.num_used] = val1._storage[:, :, indices]

    def update_threshold(self, min_threshold):
        # get the remaining indices where lb <= min_threshold
        min_threshold = torch.tensor(min_threshold, device=self.lb.storage.device)
        lb = getattr(self, 'lb').get_all()
        num_used = self.__len__()
        if self.__len__() == 0 or lb.max() < min_threshold:
            self.threshold.storage._storage[:num_used] = min_threshold
            return
        remaining_index = torch.where(
            (lb.detach() < min_threshold).all(1)
        )[0]
        # update num_used and the storage
        remaining_len = len(remaining_index)
        print(f'using {min_threshold.item()} as threshold, max lb is {lb.max().item()}')
        print(f'Remain {remaining_len} from {num_used} domains.')
        for item in self.item_list:
            getattr(self, item).update(remaining_index)
        if self.use_alpha:
            for key0 in self.alpha.keys():
                for key1 in self.alpha[key0].keys():
                    self.alpha[key0][key1].num_used = remaining_len
        if self.use_alpha:
            for key0 in self.alpha.keys():
                for key1 in self.alpha[key0].keys():
                    self.alpha[key0][key1]._storage[:remaining_len] = \
                            self.alpha[key0][key1]._storage[remaining_index]
        # update threshold with min_threshold
        getattr(self, "threshold").storage._storage[:remaining_len] = min_threshold

    def update_adv(self, attack_examples, upper_bounds):
        if not (self.store_upper_bound and self.store_adv_example):
            return
        if self.__len__() == 0:
            return
        dm_l = getattr(self, "dm_l").get_all()
        dm_u = getattr(self, "dm_u").get_all()
        ub = getattr(self, "ub").get_all()
        adv_example = getattr(self, "adv_example").get_all()
        num_domains = self.__len__()
        attack_examples = attack_examples.view(-1, *dm_l.shape[1:]).to(dm_l.device)
        for i, example in enumerate(attack_examples):
            # Check if the example is within each subdomain
            is_within_bounds = (example >= dm_l) & (example <= dm_u)
            # Reshape to collapse all dimensions except the first
            is_within_bounds = is_within_bounds.view(num_domains, -1)
            is_within_bounds = torch.all(is_within_bounds, dim=1)

            # Find subdomains where the example is within bounds
            current_upper_bound = upper_bounds[i].item()
            if torch.any(is_within_bounds):
                match_indices = torch.where(is_within_bounds)[0]
                indices_to_update = match_indices[ub[match_indices].squeeze(1) > current_upper_bound]
                ub[indices_to_update] = current_upper_bound
                adv_example[indices_to_update] = attack_examples[i]

    def inc_inlist_round(self):
        if not self.store_inlist_round:
            return
        getattr(self, "inlist_round").storage._storage[:self.__len__()] += 1

    def show_stat(self):
        print("---- Stat for current domains in list ----")
        return show_domain_stat(getattr(self, "threshold").get_all(), getattr(self, "dm_l").get_all(), 
                                getattr(self, "dm_u").get_all(), getattr(self, "lb").get_all(),
                                getattr(self, "ub").get_all() if self.store_upper_bound else None, 
                                self.global_x_L, self.global_x_U,
                                getattr(self, "inlist_round").get_all() if self.store_inlist_round else None)

    def get_total_space_size(self):
        return calculate_search_space_size(getattr(self, "dm_l").get_all(), getattr(self, "dm_u").get_all(), 
                                            self.global_x_L, self.global_x_U)[0]

def show_domain_stat(threshold, x_L, x_U, lb, ub, global_x_L=None, global_x_U=None, inlist_round=None):
    global_x_L = global_x_L.to(x_L.device)
    global_x_U = global_x_U.to(x_U.device)
    num_domains = len(x_L)
    print(f"Number of domains: {num_domains}")
    if num_domains == 0:
        print("No stat")
        return 0
    assert torch.all(threshold == threshold[0]), \
        "Thresholds are not the same."
    print(f"Threshold: {threshold[0].item()}")
    total_space_size = 0
    if global_x_L is not None and global_x_U is not None:
        total_space_size, space_size = calculate_search_space_size(x_L, x_U, global_x_L, global_x_U)
        print(f"total space size: {total_space_size:.6f}")
    print(f"{'metric':<15} {'min':<12} {'median':<12} {'max':<12}")
    if global_x_L is not None and global_x_U is not None:
        print(f"{'space':<15} {space_size.min().item():<12.6f} {space_size.median().item():<12.6f} {space_size.max().item():<12.6f}")
    if inlist_round is not None:
        print(f"{'inlist_round':<15} {inlist_round.min().item():<12.6f} {inlist_round.median().item():<12.6f} {inlist_round.max().item():<12.6f}")
    
    # print(f"{'x_L':<8} {x_L.min().item():<12.6f} {x_L.mean().item():<12.6f} {x_L.max().item():<12.6f}")
    # print(f"{'x_U':<8} {x_U.min().item():<12.6f} {x_U.mean().item():<12.6f} {x_U.max().item():<12.6f}")
    print(f"{'lb':<15} {lb.min().item():<12.6f} {lb.mean().item():<12.6f} {lb.max().item():<12.6f}")
    if ub is not None:
        print(f"{'ub':<15} {ub.min().item():<12.6f} {ub.mean().item():<12.6f} {ub.max().item():<12.6f}")
        print(f"{'ub-lb':<15} {(ub-lb).min().item():<12.6f} {(ub-lb).mean().item():<12.6f} {(ub-lb).max().item():<12.6f}")

    return total_space_size

def calculate_search_space_size(x_L, x_U, global_x_L, global_x_U):
    global_x_L = global_x_L.to(x_L.device)
    global_x_U = global_x_U.to(x_U.device)
    normed_x_L = (x_L - global_x_L) / (global_x_U - global_x_L)
    normed_x_U = (x_U - global_x_L) / (global_x_U - global_x_L)
    covered_space = normed_x_U - normed_x_L
    # [num_domain]
    space_size = covered_space.prod(-1)
    total_space_size = space_size.sum().item()
    return total_space_size, space_size

class item_storage():
    def __init__(self):
        self.storage = None
    
    def append(self, item):
        if self.storage is None:
            self.storage = get_tensor_storage(item.shape)
        self.storage.append(item.type(self.storage.dtype).to(self.storage.device))

    def pop(self, batch, device="cuda"):
        if self.storage is None:
            return None
        batch = min(len(self), batch)
        assert batch > 0, "List of InputDomain is empty; pop failed."
        item = self.storage.pop(batch).to(device=device, non_blocking=True)
        return item
    
    def remove(self, indices, device="cuda"):
        if self.storage is None:
            return
        num_used = self.storage.num_used
        remaining_indices = torch.ones(num_used, dtype=torch.bool)
        remaining_indices[indices] = False
        removed_items = self.storage._storage[indices].to(device=device, non_blocking=True)
        new_num_used = num_used - len(indices)
        self.storage._storage[:new_num_used] = self.storage._storage[:num_used][remaining_indices]
        self.storage.num_used = new_num_used
        return removed_items

    def update(self, indices):
        if self.storage is None:
            return
        new_num_used = len(indices)
        assert new_num_used <= self.storage.num_used
        self.storage._storage[: new_num_used] = self.storage._storage[indices]
        self.storage.num_used = new_num_used
    
    def get_all(self):
        if self.storage is None:
            return None
        return self.storage._storage[:self.storage.num_used]

    def __getitem__(self, idx):
        return self.storage[idx]
    
    def __len__(self):
        if self.storage is None:
            return 0
        return self.storage.num_used