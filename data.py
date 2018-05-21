from scipy.io import loadmat
import numpy as np
import pandas as pd

def convert_to_one_hot(y, c):
    return np.eye(c)[y.reshape(-1)].T

def np2one_hot(integer_encoded,type_num):
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(type_num)]
        letter[value] = 1
        onehot_encoded.append(letter)
    np_one_hot = np.array(onehot_encoded)
    return np_one_hot


class DataSet:
    """
    since index 94(0-167) has a abnormal data shape, it's removed...
    """
    mat_path = "./NASA_mill/mill.mat"

    def __init__(self):
        self.raw_data = loadmat(self.mat_path)  # type: dict
        self.mill_data = self.raw_data['mill']

    @property
    def vb_value(self):
        """
        get tool wear measured by VB
        :return: VB (166,1)
        """
        vb_data = self.mill_data["VB"].reshape(-1, 1)
        data = np.delete(vb_data, 94)
        return data

    @property
    def signal_value(self):
        """
        get signal output, and it needs conv-ing
        :return: (166,9000,6)
        """
        catalog_list = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
        signal_data = []
        for i in range(167):
            if i == 94 :
                continue
            this_catalog_data = np.array(self.mill_data["smcAC"][0][i])
            for index,catalog in enumerate(catalog_list):
                if index == 1:
                    continue
                this_catalog_data = np.concatenate((this_catalog_data,np.array(self.mill_data[catalog][0][i])),axis=1)

            signal_data.append(this_catalog_data)
        return np.array(signal_data)

    def time_value(self):
        time_list = self.mill_data["time"]
        time_data = np.delete(time_list, 94)
        return time_data

    @property
    def number_value(self):
        """
        get number configuration
        :return: (166,2)
        """
        catalog_list = ["DOC", "feed","time"]
        number_value = []
        for catalog in catalog_list:
            catalog_data = self.mill_data[catalog].reshape(-1)
            catalog_data = np.delete(catalog_data, 94)
            this_catalog_data = []
            for index, value in enumerate(catalog_data):
                # print(catalog, index, value.shape)
                this_catalog_data.append(value.reshape(-1))
            number_value.append(this_catalog_data)
        return np.array(number_value).reshape(166, 2)

    @property
    def material_type(self):
        material_type_list = self.mill_data["material"]
        material_type_list = np.delete(material_type_list, 94)
        material_type = []
        for i in material_type_list:
            material_type.append(i[0][0])
        material_type = np.array(material_type)
        np_one_hot = np2one_hot(material_type-1,2)
        return np_one_hot

    @property
    def export_as_pd(self):
        catalog = ['time','DOC','feed','material','VB'] #,'smcAC','smcDC','vib_table','vib_spindle','AE_table','AE_spindle'])
        data_dict = {}
        for i in catalog:
            data_dict[i] = self.mill_data[i].reshape(-1)
            list = []
            for index,j in enumerate(data_dict[i]):
                if index == 94:
                    continue
                data = j[0][0]
                list.append(data)
            data_dict[i] = list

        return pd.DataFrame(data_dict)

    @property
    def rf_vb_value(self):
        from rf_fit_data import fit_value_by_random_forest
        return fit_value_by_random_forest()

if __name__ == '__main__':
    a = DataSet()
    signal = a.signal_value
    print('^ signal_shape',signal.shape)
    material_list = a.material_type
    print(material_list.shape)
    conf_number = a.number_value
    print(conf_number.shape)
