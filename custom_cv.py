@dataclass
class MultiGroupKFold():
    n_splits: int = field(init=True)
    groups: np.ndarray #check this line
    groups_in_train: int = field(init=True)
    random_state: int = field(init=True)
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    def split(self, X=None, y=None, groups=None):
        index_list = []
        test_group_combinations = []
        unique_groups = np.unique(self.groups)
        for group in unique_groups:
            index_list.append(np.where(self.groups == group))
        index_array = np.array(index_list, dtype=object)
        train_group_combinations = [combination for combination in \
            itertools.combinations(unique_groups, self.groups_in_train)]
        number_of_combinations = len(train_group_combinations)
        for combination in train_group_combinations:
            test_group_combinations.append(list(set(unique_groups) - \
            set(combination)))
        train_indices = []
        for i, combination in enumerate(train_group_combinations):
            current_list = []
            for group in combination:
                group_index = np.squeeze(np.where(unique_groups == group))
                current_list.append(index_list[group_index])
            train_indices.append(current_list)
        train_index_array = np.reshape(np.array(train_indices,dtype=object), \
                (number_of_combinations,-1))
        test_indices = []
        for i, combination in enumerate(test_group_combinations):
            current_list = []
            for group in combination:
                group_index = np.squeeze(np.where(unique_groups == group))
                current_list.append(index_list[group_index])
            test_indices.append(current_list)
        test_index_array = np.reshape(np.array(test_indices,dtype=object), \
            (number_of_combinations,-1))
        rng = default_rng(seed=self.random_state)
        for fold in range(self.n_splits):
            train_indices_for_fold = np.concatenate(train_index_array[fold])
            test_indices_for_fold = np.concatenate(test_index_array[fold])
            rng.shuffle(train_indices_for_fold)
            rng.shuffle(test_indices_for_fold)
            yield (train_indices_for_fold, test_indices_for_fold)

@dataclass
class ShuffleMultiGroupKFold():
    n_splits: int = field(init=True)
    groups: np.ndarray
    train_proportion: float = field(init=True)
    test_proportion: float = field(init=True)
    groups_in_train: int = field(init=True)
    random_state: int = field(init=True)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        index_list = []
        test_group_combinations = []
        unique_groups = np.unique(self.groups)
        #index_list is a list of lists with one list for each category
        #the indices are the row numbers of groups
        for group in unique_groups:
            #index_list.append(   self.groups.index[np.where(self.groups == group)]    )
            index_list.append(np.where(self.groups == group))
        index_array = np.array(index_list, dtype=object)
        #train_group_combinations is a list of tuples that has valid groups
        #in train, so has grops_in_train in each tuple, for example ('A', 'B',
        #'C')
        train_group_combinations = [
            combination
            for combination in itertools.combinations(
                unique_groups, self.groups_in_train)]
        number_of_combinations = len(train_group_combinations)
        #Test group combinations has the inverse of what's in
        #train_group_combinations
        for combination in train_group_combinations:
            test_group_combinations.append(
                list(set(unique_groups) - set(combination)))
        train_indices = []
        for i, combination in enumerate(train_group_combinations):
            current_list = []
            for group in combination:
                #group_index is the index of that group in index_list
                group_index = np.squeeze(np.where(unique_groups == group))
                current_list.append(index_list[group_index])
            train_indices.append(current_list)
        train_index_array = np.reshape(
            np.array(train_indices, dtype=object),
            (number_of_combinations, -1))
        test_indices = []
        for i, combination in enumerate(test_group_combinations):
            current_list = []
            for group in combination:
                group_index = np.squeeze(np.where(unique_groups == group))
                current_list.append(index_list[group_index])
            test_indices.append(current_list)
        test_index_array = np.reshape(
            np.array(test_indices, dtype=object),
            (number_of_combinations, -1))
        rng = default_rng(seed=self.random_state)
        for fold in range(self.n_splits):
            #our train test arrays are aligned, so we have to take from the
            #same position to respect the separation of the groups
            random_index_position = rng.integers(
                low=0, high=number_of_combinations)
            train_indices_for_fold = np.concatenate(
                train_index_array[random_index_position])
            test_indices_for_fold = np.concatenate(
                test_index_array[random_index_position])
            rng.shuffle(train_indices_for_fold)
            rng.shuffle(test_indices_for_fold)
            #rng = default_rng(self.random_state)
            #rng2 = default_rng(self.random_state)
            yield (rng.choice(train_indices_for_fold, \
            size=int(len(train_indices_for_fold) * self.train_proportion), \
            replace=False),
                   rng.choice(test_indices_for_fold, \
                   size=int(len(test_indices_for_fold) * self.test_proportion),\
                   replace=False),
                   )
