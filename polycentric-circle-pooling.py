import numpy as np
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras import backend
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.framework import tensor_shape


class PCP(Layer):
    """Polycentric Concentric Circle pooling layer for 2D inputs.
    See Polycentric Circle Pooling in Deep Convolutional Network for High-Resolution Remote Sensing Recognition (under review),
    K. Qi, C. Hu, C. Yang, Q. Guan
    # Arguments
        circle_number: int
            Number of annular subregions to use for each concentric circle pooling with different centers. 
        center_xy: array
            Location of centers to use. The dimension of the array is (n, 2), which n is the number of centers. Each row value represent the location with the origin of feature map center.  For example [[-1,0], [0, 0], [1, 0]] would be 3 concentric circle poolings, and a annular subregion is added for center [-1,0] and [1, 0], so 3x`circle_number`+2 outputs per feature map.
        pool_mode: str
            max pooling for each annular subregions     if pool_mode='max'
            average pooling for each annular subregions if pool_mode='avg'
    # Input shape
        4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.        
    # Output shape
        2D tensor with shape:
        `(samples, channels * (circle_number*len(center_xy)+np.sum(np.max(np.abs(center_xy))))`
    """

    def __init__(self, circle_number, center_xy=[[0,0]], pool_mode='max', **kwargs):
        """
        Initialize the class
        :param circle_count: int, number of circle
        :param center_xy: array, location of centers
        :param pool_mode: string, `"max"` or `"avg"`.
        :param kwargs:
        """
        data_format = backend.image_data_format()
        data_format = conv_utils.normalize_data_format(data_format)
        assert data_format == 'channels_last'            

        self.circle_number = circle_number
        self.center_xy = center_xy

        num_addition = np.sum(np.max(np.abs(np.array(self.center_xy, np.int8)), axis=1))
        self.num_outputs_per_channel = sum(circle_number)*len(self.center_xy) + num_addition

        self.pool_mode = pool_mode

        super(PCP, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[3]
        
    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape([input_shape[0], 
            self.nb_channels * self.num_outputs_per_channel])

    def call(self, input):
        input_shape = backend.shape(input)

        num_rows = input_shape[1]
        num_cols = input_shape[2]

        row_length = backend.cast(num_rows, 'float32') / (2*self.circle_number)
        col_length = backend.cast(num_cols, 'float32') / (2*self.circle_number)

        outputs = []
        pool_circle = []
        for jy in range(self.circle_number*2): 
            for ix in range(self.circle_number*2):
                x1 = ix * col_length
                x2 = ix * col_length + col_length
                y1 = jy * row_length
                y2 = jy * row_length + row_length

                x1 = backend.cast(backend.round(x1), 'int32')
                x2 = backend.cast(backend.round(x2), 'int32')
                y1 = backend.cast(backend.round(y1), 'int32')
                y2 = backend.cast(backend.round(y2), 'int32')

                new_shape = [input_shape[0], y2 - y1, x2 - x1, input_shape[3]]

                x_crop = input[:, y1:y2, x1:x2, :]
                xm = backend.reshape(x_crop, new_shape)
                if self.pool_mode == 'avg':
                    pooled_val = backend.mean(xm, axis=(1, 2))
                else:
                    pooled_val = backend.max(xm, axis=(1, 2))
                pool_circle.append(backend.reshape(xm, (input_shape[0], -1, input_shape[3])))

        circle_index = self._circle_index(self.circle_number)
        for cidx in circle_index:
            circle_val = [pool_circle[idx] for idx in cidx]
            if self.pool_mode == 'avg':
                pooled_val = backend.mean(backend.concatenate(circle_val, axis=1), axis=1)
            else:
                pooled_val = backend.max(backend.concatenate(circle_val, axis=1), axis=1)

            outputs.append(pooled_val)

        outputs = backend.concatenate(outputs)
        outputs = backend.reshape(outputs, (input_shape[0], self.nb_channels*self.num_outputs_per_channel))

        return outputs

    def get_config(self):
        config = {
            'circle_count': self.circle_count,
            'center_xy': self.center_xy,
            'pool_mode': self.pool_mode}
        base_config = super(PCP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def _circle_index(self, circle_number):
        indexs = []
        row_cols = self.center_xy if circle_number > 1 else [[0, 0]]
        for jrow, jcol in row_cols:
            for ic in range(circle_number + max(abs(jrow), abs(jcol))):
                head, tail = circle_number-1-ic, circle_number+1+ic

                row_head = max(head+jrow, 0)
                row_tail = min(tail+jrow, 2*circle_number)
                col_head = max(head+jcol, 0)
                col_tail = min(tail+jcol, 2*circle_number) 

                idx = []
                if head+jrow >= 0:
                    idx.extend([(head+jrow)*2*circle_number+icol \
                        for icol in range(col_head, col_tail)])
                if tail+jrow <= 2*circle_number:
                    idx.extend([(tail+jrow-1)*2*circle_number+icol \
                        for icol in range(col_head, col_tail)])
                
                if head+jcol >= 0:
                    idx.extend([irow*2*circle_number+head+jcol \
                        for irow in range(row_head, row_tail)])
                if tail+jcol <= 2*circle_number:
                    idx.extend([irow*2*circle_number+tail+jcol-1 \
                        for irow in range(row_head, row_tail)])

                indexs.append(set(idx))
        return indexs
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)  
