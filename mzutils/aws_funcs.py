import os
import hashfile


def hash_compare_two_files(file_path1, file_path2):
    """check whether the hash of two files are the same.

    Args:
        file_path1 (str): path to file 1, can be s3 path
        file_path2 (str): path to file 2, can be s3 path
        
    Returns:
        tuple of bool, f1_hash, f2_hash: 
        whether the hash of two files are the same, and their hashes respectively.
    """
    if file_path1.startswith("s3://") or file_path2.startswith("s3://"):
        import s3fs
        import tempfile
        with tempfile.TemporaryDirectory() as temp1_dir:
            with tempfile.TemporaryDirectory() as temp2_dir:
                s3 = s3fs.S3FileSystem()
                s3.read_timeout=120
                s3.connect_timeout=120
                s3.retries={'max_attempts': 5, 'mode': 'standard'}
                if file_path1.startswith("s3://"):
                    s3.get(file_path1, temp1_dir, recursive=True)
                    file1 = os.path.join(temp1_dir, os.path.basename(file_path1))
                else:
                    file1 = file_path1
                    
                if file_path2.startswith("s3://"):
                    s3.get(file_path2, temp2_dir, recursive=True)
                    file2 = os.path.join(temp2_dir, os.path.basename(file_path2))
                else:
                    file2 = file_path2
            
                f1_hash = hashfile(file1)
                f2_hash = hashfile(file2)
    else:
        f1_hash = hashfile(file1)
        f2_hash = hashfile(file2)
    
    return f1_hash == f2_hash, f1_hash, f2_hash
