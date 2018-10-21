import numpy as np
import os
import time
import sys
import yaml
import logging
from subprocess import Popen, PIPE
import torch.autograd, torch.nn
'''
    Utility functions
    Taken from keras code : https://keras.io/
'''


def print_metrics(metrics, fp=None):
    metric_str = ""
    for metric in metrics:
        metric_str += '\t%s: %.4f' % (metric, metrics[metric])
    if fp is None:
        print(metric_str)
    else:
        with open(fp, 'wb') as f:
            f.write(metric_str)


def setup_logger(loglevel, logfile=None):
    """Sets up the logger

    Arguments:
        loglevel (str): The log level (INFO|DEBUG|..)
        logfile Optional[str]: Add a file handle

    Returns:
        None
    """
    numeric_level = getattr(logging, loglevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logger = logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: %(message)s',
        level=numeric_level, stream=sys.stdout)
    if logfile is not None:
        fmt = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
        logfile_handle = logging.FileHandler(logfile, 'w')
        logfile_handle.setFormatter(fmt)
        logger.addHandler(logfile_handle)


def setup_output_dir(output_dir, config, loglevel):
    """
        Takes in the output_dir. Note that the output_dir stores each run as run-1, ....
        Makes the next run directory. This also sets up the logger
        A run directory has the following structure
        run-1:
            |_ best_model
                     |_ model_params_and_metrics.tar.gz
                     |_ validation paths.txt
            |_ last_model_params_and_metrics.tar.gz
            |_ config.yaml
            |_ githash.log of current run
            |_ gitdiff.log of current run
            |_ logfile.log (the log of the current run)
        This also changes the config, to add the save directory
    """
    make_directory(output_dir, recursive=True)
    last_run = -1
    for dirname in os.listdir(output_dir):
        if dirname.startswith('run-'):
            last_run = max(last_run, int(dirname.split('-')[1]))
    new_dirname = os.path.join(output_dir, 'run-%d' % (last_run + 1))
    make_directory(new_dirname)
    best_model_dirname = os.path.join(new_dirname, 'best_model')
    make_directory(best_model_dirname)
    config_file = os.path.join(new_dirname, 'config.yaml')
    config['data_params']['save_dir'] = new_dirname
    write_to_yaml(config_file, config)
    # Save the git hash
    process = Popen('git log -1 --format="%H"'.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('utf-8').strip('\n').strip('"')
    with open(os.path.join(new_dirname, "githash.log"), "w") as fp:
        fp.write(stdout)
    # Save the git diff
    process = Popen('git diff'.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    with open(os.path.join(new_dirname, "gitdiff.log"), "w") as fp:
        stdout = stdout.decode('utf-8')
        fp.write(stdout)
    # Set up the logger
    logfile = os.path.join(new_dirname, 'logfile.log')
    setup_logger(loglevel, logfile)
    return new_dirname, config


def read_from_yaml(filepath):
    with open(filepath, 'r') as fd:
        data = yaml.load(fd)
    return data


def write_to_yaml(filepath, data):
    with open(filepath, 'w') as fd:
        yaml.dump(data=data, stream=fd, default_flow_style=False)


def make_directory(dirname, recursive=False):
    if recursive:
        directories = dirname.split('/')
        root = directories[0]
        make_directory(root, recursive=False)
        for directory in directories[1:]:
            root = os.path.join(root, directory)
            make_directory(root, recursive=False)
    else:
        try:
            os.makedirs(dirname)
        except OSError:
            if not os.path.isdir(dirname):
                raise


def disp_params(params, name):
        print_string = "{0}".format(name)
        for param in params:
            print_string += '\n\t%s: %s' % (param, str(params[param]))
        # print(print_string)
        logger = logging.getLogger()
        logger.info(print_string)


def convert2gpu(var, gpu):
    return var.cuda() if gpu else var


def getnumpy(tensor, gpu):
    return tensor.cpu().data.numpy() if gpu else tensor.data.numpy()


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads each sequence to the same length (length of the longest sequence).

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)

    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        if target is None:
            target = -1
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, force=False):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        """
        values = values or []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return

            prev_total_width = self.total_width
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')

            if self.target is not -1:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
                bar = barstr % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
                sys.stdout.write(bar)
                self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target and self.target is not -1:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * ' ')

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write('\n')

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)


def to_cuda(t, gpu):
    return t.cuda() if gpu else t

def to_numpy(t, gpu):
    """
        Takes in a Variable, and returns numpy
    """
    ret = t.data if isinstance(t, (torch.autograd.Variable, torch.nn.Parameter)) else t
    ret = ret.cpu() if gpu else ret  # this brings it back to cpu
    return ret.numpy()