import os, subprocess, logging, json
import numpy as np
import pandas as pd
from itertools import permutations
from tqdm import tqdm
from collections import defaultdict
from sys import maxsize
logging.basicConfig(level=logging.INFO)

class PartitionBase:

    def prepare(self, exe_path, board_ip_addr, board_path, result_dir, board_pwd='khadas'):
        self.exe_path = exe_path
        self.board_ip_addr = board_ip_addr
        self.board_path = board_path
        self.board_pwd = board_pwd
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        # get number of nodes in the graph
        self.n = self._get_number_of_nodes_from_board()

    def run(self):        
        # search alogrithm for the best partition
        ## brute force search for partition base
        self._brute_force_search()

    def done(self):
        # organize results
        self._organize_and_save_results()
    
    def _copy_file_to_board(self, src_file, dst_path):
        subprocess.run(f"sshpass -p khadas scp {src_file} root@{self.board_ip_addr}:{dst_path}", shell=True)
    
    def _copy_file_from_board(self, src_file, dst_path):
        subprocess.run(f"sshpass -p khadas scp root@{self.board_ip_addr}:{src_file} {dst_path}", shell=True)
    
    def _run_inference_with_profiling(self):
        pass

    def _get_number_of_nodes_from_board(self):
        # 1. push exe file to board 
        self._copy_file_to_board(self.exe_path, self.board_path)
        # 2. run model inference with random parameter settings
        shell_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shell', 'khadas_armcl_test.sh')
        log_path = os.path.join(self.result_dir, 'profiling.log')
        shell_params = {
            1: self.board_ip_addr,
            2: self.board_path,
            3: os.path.basename(self.exe_path),
            4: 'B-L-G',
            5: '2',
            6: '6',
            7: log_path
        }
        subprocess.call([f'chmod +x {shell_script_path}'], shell=True)
        subprocess.call([shell_script_path] + [shell_params[i] for i in shell_params.keys()])
        # cmd = f"""sshpass -p khadas ssh root@{self.board_ip_addr};
        # cd {self.board_path}; 
        # export LD_LIBRARY_PATH={self.board_path};
    	# ./{os.path.basename(self.exe_path)} --threads=4 --threads2=2 --n=1 --total_cores=6 partition_point=2 partition_point2=6 --order=B-L-G > {self.board_path}/${os.path.basename(self.exe_path)}.log;"""

        # 3. parse log to get node number
        lines = open(log_path, 'r').readlines()
        n = None
        for line in lines:
            if 'Total parts' in line:
                n = int(line.split(':')[-1].strip())
        assert n is not None, \
            f"Cannot find number of nodes information in {log_path}"
        logging.info(f"Number of nodes in graph is {n}")
        return n

    def _brute_force_search(self):
        shell_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shell', 'khadas_armcl_test.sh')
        log_path = os.path.join(self.result_dir, 'profiling.log')
        res = defaultdict(list)
        best_fr, best_fl = 0, maxsize
        best_fr_sol, best_fl_sol = {}, {}
        for sorder in tqdm(list(permutations(['B', 'L', 'G']))):
            for i in range(1, self.n):
                for j in range(i + 1, self.n + 1):
                    logging.info(f'Cut1 = {i}; Cut2 = {j}...')
                    shell_params = {
                        1: self.board_ip_addr,
                        2: self.board_path,
                        3: os.path.basename(self.exe_path),
                        4: '-'.join(sorder),
                        5: str(i),
                        6: str(j),
                        7: log_path
                    }
                    subprocess.call([f'chmod +x {shell_script_path}'], shell=True)
                    subprocess.call([shell_script_path] + [shell_params[i] for i in shell_params.keys()])
                    
                    # parse frame rate and latency
                    res['sorder'].append(sorder)
                    res['cut1'].append(i)
                    res['cut2'].append(j)
                    lines = open(log_path, 'r').readlines()
                    fr, fl = None, None
                    for line in lines:
                        if 'Frame rate' in line:
                            fr = float(line.split(': ')[-1].split(' ')[0])
                        elif 'Frame latency' in line:
                            fl = float(line.split(': ')[-1].split(' ')[0])
                    res['fr'].append(fr)
                    res['fl'].append(fl)

                    if fr > best_fr:
                        best_fr = fr
                        best_fr_sol = {
                            'order': '-'.join(sorder),
                            'cut1': i,
                            'cut2': j
                        }
                    if fl < best_fl:
                        best_fl = fl
                        best_fl_sol = {
                            'order': '-'.join(sorder),
                            'cut1': i,
                            'cut2': j
                        }
        res = pd.DataFrame(res)
        # save results 
        ## csv for search results
        csv_path = os.path.join(self.result_dir, 'search_result.csv')
        res.to_csv(csv_path, index=False)
        ## solutions for the best frame rate and frame latency
        solution_path = os.path.join(self.result_dir, 'solution.json')
        json.dump(
            {'best_fr_solution': best_fr_sol,
             'best_fl_solution': best_fl_sol}, 
            open(solution_path, 'w'),
            indent=4
        )
        logging.info(f'The best frame rate solution is {best_fr_sol}\nThe best latency solution is {best_fl_sol}\n')

if __name__ == '__main__':
    pipe_all_example_dir = '/home/pzan/compiler/ARMCL-Pipe-All/build/examples'
    example_fname = 'graph_alexnet_all_pipe_sync'
    result_dir = os.path.join('/home/pzan/compiler/result', example_fname)
    partition = PartitionBase()
    partition.prepare(
        exe_path=os.path.join(pipe_all_example_dir, example_fname),
        board_ip_addr='192.168.1.2',
        board_path='/usr/test/armcl',
        result_dir=result_dir
    )
    partition.run()