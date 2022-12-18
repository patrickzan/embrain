import os, subprocess
import numpy as np

class PartitionBase:

    def prepare(self, exe_path, board_ip_addr, board_path, board_pwd='khadas'):
        self.exe_path = exe_path
        self.board_ip_addr = board_ip_addr
        self.board_path = board_path
        self.board_pwd = board_pwd

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
        
        # cmd = f"""sshpass -p khadas ssh root@{self.board_ip_addr};
        # cd {self.board_path}; 
        # export LD_LIBRARY_PATH={self.board_path};
    	# ./{os.path.basename(self.exe_path)} --threads=4 --threads2=2 --n=1 --total_cores=6 partition_point=2 partition_point2=6 --order=B-L-G > {self.board_path}/${os.path.basename(self.exe_path)}.log;"""

    def _brute_force_search(self):
        pass

    def _organize_and_save_results():
        pass