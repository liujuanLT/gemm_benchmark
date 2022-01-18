import argparse
from pathlib import Path
import os
import csv

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import lt_sdk.config.light_config as light_config
import lt_sdk.verification.workloads.tests.ALU_vec_matmul as ALU
from lt_sdk.api import api
from lt_sdk.graph.import_graph import tf_graph_def_importer
from lt_sdk.visuals import sim_result_to_trace
fre = 1E6  # clock per ms

def _get_tf_DPMP_graph(x, A, n_iter):
    """
    Create and return the tensorflow graph def object for a test DPMP matmul worklaod

    Args:
        x: input vector
        A: input weight matrix
        n_iter: number of desired iterations

    Return:
        Tensorflow graph_def object
    """
    g = tf.Graph()
    with g.as_default():
        x_t = tf.compat.v1.placeholder(tf.float32, shape=x.shape, name="x")
        A_t = tf.constant(A.astype(np.float32), name="A")
        for _ in range(n_iter):
            out_t = tf.matmul(x_t, A_t)
        #for _ in range(1, n_iter):
         #i#   out_t = tf.matmul(out_t, A_t)

    return g.as_graph_def()

def plot_fig(matrix_sizes, DPMP_runtimes, comp=None, comp_label=None, output_path=None):
    fig, (ax, logax) = plt.subplots(ncols=2)
    fig.set_figwidth(14)
    fig.set_figheight(4.8)
    ax.semilogx(matrix_sizes, comp, label=comp_label)
    ax.semilogx(matrix_sizes, DPMP_runtimes, label="DPMP")
    ax.set(xlabel="Matrix Size",
           ylabel="Total ms",
           title=f"{comp_label} vs DPMP Timing (linlog)")
    ax.grid()
    ax.legend()

    logax.loglog(matrix_sizes, comp, label=comp_label)
    logax.loglog(matrix_sizes, DPMP_runtimes, label="DPMP")
    logax.set(xlabel="Matrix Size",
              ylabel="Total ms",
              title=f"{comp_label} vs DPMP Timing (loglog)")
    logax.grid()
    logax.legend()
    out_file = os.path.join(output_path, f"{comp_label}vsDPMP_figure.png")
    fig.savefig(out_file)
    print(f'saved to {out_file}')

def plot_table(matrix_sizes, DPMP_runtimes, comp, comp_label, output_path=None):
    tab, tabax = plt.subplots()
    tab.set_figheight(1.5)
    tabax.axis("off")
    tabax.table(cellText=[comp,
                          DPMP_runtimes],
                rowLabels=[comp_label,
                           "DPMP"],
                colLabels=matrix_sizes,
                loc="center")
    out_file = os.path.join(output_path, f"{comp_label}vsDPMP_table.png")
    tab.savefig(out_file)
    print(f'saved to {out_file}')

def save_to_csv(comp_label, lines, output_path=None):
    out_file = os.path.join(output_path, f"{comp_label}vsDPMP_figure.csv")
    with open(out_file, 'w') as new_file: 
        csv_writer = csv.writer(new_file, delimiter=' ')
        for line in lines:
            csv_writer.writerow(line)
    print(f'saved to {out_file}')

def ALU_vs_DPMP_performance(matrix_sizes, n_iter, output_dir, ignore_moves, m_spec=None, n_spec=None, k_spec=None, comp=None, comp_label='GPU'):
    DPMP_runtimes = []
    output_path = Path(output_dir)
    trace_path = output_path / Path("traces/")
    trace_path.mkdir(parents=True, exist_ok=True)

    for var_size in matrix_sizes:
        m = m_spec if m_spec else var_size
        n = n_spec if n_spec else var_size
        k = k_spec if k_spec else var_size
        x = np.random.rand(m, k).astype(np.float32)
        A = np.random.rand(k, n).astype(np.float32)

        # --- Compute using DPMP on Dagger (int32)
        DPMP_config = light_config.get_default_config(
            hw_cfg=light_config.hardware_configs_pb2.DAGGER,
            graph_type=light_config.graph_types_pb2.TFGraphDef)
        DPMP_graph = tf_graph_def_importer.ImportTFGraphDef(
            None,
            DPMP_config.sw_config,
            graph_def=_get_tf_DPMP_graph(x,
                                         A,
                                         n_iter)).as_light_graph()
        light_graph = api.transform_graph(DPMP_graph, DPMP_config)
        execution_stats = api.run_performance_simulation(light_graph, DPMP_config)
        if ignore_moves:
            max_clock = 0
            for instruction in execution_stats.instructions:
                operation = instruction.instruction.node.WhichOneof("node")
                if operation != "move":
                    max_clock = max(instruction.start_clk + instruction.duration_clks,
                                    max_clock)
            DPMP_runtimes.append(max_clock)
        else:
            DPMP_runtimes.append(execution_stats.total_clocks)

        filename = "output_DPMP_matrix_m{}xn{}xk{}_niter{}.trace".format(m, n, k, n_iter)
        save_path = trace_path / filename
        sim_result_to_trace.instruction_trace(save_path,
                                              execution_stats,
                                              DPMP_config.hw_specs,
                                              DPMP_config.sim_params)
    DPMP_runtimes = [ v/fre for v in DPMP_runtimes ] # clocks to duration
    plot_fig(matrix_sizes, DPMP_runtimes, comp, comp_label, output_path=output_path)
    plot_table(matrix_sizes, DPMP_runtimes, comp, comp_label, output_path=output_path)
    lines = []
    s = f'm, n, k, {comp_label}, DPMP'
    lines.append(['m', 'n', 'k', comp_label, 'DPMP'])
    print(s)
    for i in range(len(matrix_sizes)):
        var_size = matrix_sizes[i]
        m = m_spec if m_spec else var_size
        n = n_spec if n_spec else var_size
        k = k_spec if k_spec else var_size
        s = f'{m}, {n}, {k}, {comp[i]}, {DPMP_runtimes[i]}'
        lines.append([m, n, k, comp[i], DPMP_runtimes[i]])
        print(s)
    save_to_csv(comp_label, lines, output_path=output_path)

def var_k_test_1(args):
    k_list = [1024] # [list(range(1008, 1025))]
    m = n = 1024
    subsavedir = os.path.join(args.output_dir, f'mn{m}')
    ALU_vs_DPMP_performance(k_list,
                        args.n_iter,
                        subsavedir,
                        args.ignore_moves,
                        m_spec = m,
                        n_spec = n,
                        comp = [0.04],
                        comp_label='GPU')

def var_mn_test(args):
    # mn_list = [32]
    mn_list = [2**n for n in range(5, 14)]
    k = 4096
    subsavedir = os.path.join(args.output_dir, f'k{k}')
    ALU_vs_DPMP_performance(mn_list,
                        args.n_iter,
                        subsavedir,
                        args.ignore_moves,
                        k_spec = k,
                        comp = [1.3*pow(2,-6), 1.3*pow(2,-6), 1.3*pow(2,-6), 1.3*pow(2,-6), 
                        1.8*pow(2,-6) , 1.8*pow(2,-5) , 1.3*pow(2,-3), 1.1*pow(2,-1), pow(2, 1)],
                        comp_label = 'GPU')        

def var_k_test_2(args):
    k_list = [2**n for n in range(5, 14)]
    m = n = 4096
    subsavedir = os.path.join(args.output_dir, f'mn{m}')
    ALU_vs_DPMP_performance(k_list,
                        args.n_iter,
                        subsavedir,
                        args.ignore_moves,
                        m_spec = m,
                        n_spec = n,
                        comp = [1.2*pow(2, -5), 1.4*pow(2, -5), 1.5*pow(2, -5), 1.8*pow(2, -5), 
                        1.4*pow(2, -4), 1.3*pow(2, -3), 1.2*pow(2, -2), 1.1*pow(2, -1), 1.3*pow(2, 0)])

def native_test(args):
    kn_list = [2**n for n in range(6,12)]
    m = 1
    subsavedir = os.path.join(args.output_dir, f'm{m}')
    ALU_vs_DPMP_performance(kn_list,
                        args.n_iter,
                        subsavedir,
                        args.ignore_moves,
                        m_spec = m,
                        comp = [0] * len(kn_list),
                        comp_label='unknown')
   
   
def native_test_2(args):
    m_list = [1] + [2**n for n in range(6,12)]
    k = 64
    n = 256
    subsavedir = os.path.join(args.output_dir, f'k{k}n{n}')
    ALU_vs_DPMP_performance(m_list,
                        args.n_iter,
                        subsavedir,
                        args.ignore_moves,
                        k_spec = k,
                        n_spec = n,
                        comp = [0.0018, 0.0210, 0.0, 0.0230, 0.0255, 0.0266, 0.0352],
                        comp_label='CPU')   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the performance of \
            Vector-Matrix multiply on ALU vs DPMP")
    parser.add_argument("--output_dir",
                        action="store",
                        type=str,
                        help="Custom output directory for \
                            traces, figures and table",
                        default=".")
    parser.add_argument("--sizes",
                        action="store",
                        nargs="+",
                        type=int,
                        help="The list of matrix sizes to be tested",
                        default=[2**n for n in range(6,
                                                     12)])
    parser.add_argument("--n_iter",
                        action="store",
                        type=int,
                        help="The number of iterations to be tested",
                        default=1)
    parser.add_argument("--n_vecs",
                        action="store",
                        type=int,
                        help="The number of simoultaneous input vectors to be tested",
                        default=1)

    parser.add_argument("--ignore_moves",
                        action="store_true",
                        help="Ignore move times in the total clock count")

    args = parser.parse_args()
    args.n_iter = 1
    args.ignore_moves = True
    args.output_dir = f'/home/jliu/codes/data/niter{args.n_iter}_ignoremoves{args.ignore_moves}'
    # var_k_test_1(args)
    # var_mn_test(args)
    # var_k_test_2(args)
    # native_test(args)
    native_test_2(args)
