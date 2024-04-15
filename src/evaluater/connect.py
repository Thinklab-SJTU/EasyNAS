import os
import time
import ast
import paramiko

# Connect to Computer B
def build_sshclient(host, username, password=None, pkey=None, port=22):
    assert password is not None or pkey is not None
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if pkey and os.path.exists(pkey):
        private_key = paramiko.RSAKey.from_private_key_file(pkey)
        ssh.connect(host, port=port, username=username, pkey=pkey)
    else:
        ssh.connect(host, port=port, username=username, password=password)
    return ssh


def fetch_info(ssh, cmd, end_identifier, output_parser=None):
    ssh_shell = ssh.invoke_shell()

    TIME_TH = 90 # seconds
    t = time.time()
    ssh_shell.send("echo hello \n")
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    output = stdout.read().decode()
    print(output)
    print(stderr.read().decode())

#    ssh_shell.send(cmd)
#    output = ""
#    newout = ""
#    while not ssh_shell.exit_status_ready() and time.time() - t < TIME_TH:
#        if ssh_shell.recv_ready():
#            newout = ssh_shell.recv(1024).decode()
#            print(newout)
#            output += newout
#        if time.time() - t > 4 and end_identifier in newout:
#            break
        
    #print(output)
    # Execute the command on Computer B and get the output
    # stdin, stdout, stderr = ssh.exec_command("python test.py", timeout=60)
    # output = stdout.read().decode()
    # Close the SSH connection
    #ssh.close()

    # Process the output and do something with it
    try:
        if output_parser is not None:
            info = output_parser(output)
        else: info = output
    except Exception as e:
        print('An error occured:', e)
        return None
    else:
        return info

def fetch_info_rk3588(ssh, cmd):
    end_identifier = '(base)'
    def output_parser(output):
        '''
        Input: output of the command
        Output:
            latency, memory
        '''
        latency, memory = 0, 0
        for line in output.splitlines():
            if 'latency' in line:
                latency = float(line.split('latency: ')[-1]) * 1000 # convert second to micro-second
            if 'Total Memory' in line:
                memory = float(line.split('Total Memory: ')[-1].split('MiB')[0])

        if latency == 0 or memory == 0:
            raise Exception('Failed to parse output: \n{}'.format(output))
        return {'latency': latency, 'memory': memory}
    info = fetch_info(ssh, cmd, end_identifier, output_parser)
    return info

