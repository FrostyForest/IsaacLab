import os
import shutil


def delete_small_dirs_recursively(root_folder, size_limit_kb=5, dry_run=True):
    """
    递归地删除指定文件夹中总大小小于给定阈值的文件夹。

    Args:
        root_folder (str): 要扫描的根文件夹路径。
        size_limit_kb (int, optional): 大小阈值 (单位: KB)。默认为 5。
        dry_run (bool, optional): 是否为演习模式。
                                True: 只打印将要删除的文件夹，不实际删除。
                                False: 实际执行删除操作。
                                默认为 True。
    """
    # 将 KB 转换为 Bytes
    size_limit_bytes = size_limit_kb * 1024

    # 规范化路径，以确保跨平台兼容性和准确的路径比较
    root_folder = os.path.abspath(root_folder)

    if not os.path.isdir(root_folder):
        print(f"错误：提供的路径 '{root_folder}' 不是一个有效的文件夹。")
        return

    if dry_run:
        print("--- 演习模式 (Dry Run) ---")
        print("将查找并列出所有小于 " + str(size_limit_kb) + " KB 的文件夹，但不会实际删除它们。")
    else:
        print("--- !!! 真实删除模式 !!! ---")
        print("将永久删除所有小于 " + str(size_limit_kb) + " KB 的文件夹。")
        # 在真实删除前给出用户确认机会
        confirm = input(f"你确定要在 '{root_folder}' 中执行此操作吗？ (yes/no): ")
        if confirm.lower() != "yes":
            print("操作已取消。")
            return

    # 用于存储已计算的目录大小，避免重复计算
    dir_sizes = {}
    deleted_dirs_count = 0

    # 关键：使用 topdown=False 进行自下而上的遍历
    # 这确保了我们先处理子目录，再处理父目录
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        # 1. 计算当前目录中所有文件的总大小
        current_dir_file_size = sum(os.path.getsize(os.path.join(dirpath, f)) for f in filenames)

        # 2. 加上所有子目录的大小（这些大小已在之前的循环中计算并存储）
        current_dir_subdirs_size = sum(dir_sizes.get(os.path.join(dirpath, d), 0) for d in dirnames)

        # 3. 计算当前目录的总大小
        total_dir_size = current_dir_file_size + current_dir_subdirs_size

        # 4. 存储当前目录的总大小，供上层目录使用
        dir_sizes[dirpath] = total_dir_size

        # 5. 判断是否要删除
        # a. 确保不是根目录本身
        # b. 确保大小小于阈值
        if dirpath != root_folder and total_dir_size < size_limit_bytes:
            deleted_dirs_count += 1
            # 格式化大小以便阅读
            size_str = f"{total_dir_size / 1024:.2f} KB" if total_dir_size > 0 else f"{total_dir_size} Bytes"

            action_prefix = "[演习] 将删除" if dry_run else "[正在删除]"
            print(f"{action_prefix}: '{dirpath}' (总大小: {size_str})")

            if not dry_run:
                try:
                    # 使用 shutil.rmtree() 删除整个目录树
                    shutil.rmtree(dirpath)
                except OSError as e:
                    print(f"  -> 错误: 无法删除 '{dirpath}'. 原因: {e}")

    print("\n--- 操作完成 ---")
    if dry_run:
        print(f"在演习模式下，共找到 {deleted_dirs_count} 个符合条件的文件夹。")
    else:
        print(f"共删除了 {deleted_dirs_count} 个文件夹。")


# --- 如何使用 ---
if __name__ == "__main__":
    # 1. 设置你要扫描的文件夹路径
    # 重要：请务必将下面的路径更改为你自己的目标文件夹！
    # 例如: "C:/Users/YourUser/Downloads/test_folder" 或 "/home/user/documents/cleanup"
    target_directory = "/home/linhai/code/IsaacLab/runs"

    # 2. 第一次运行时，强烈建议使用 dry_run=True (默认值)
    # 这会让你看到哪些文件夹会被删除，而不会真的删除它们
    print("第一步：执行演习，检查哪些文件夹会被删除。")
    delete_small_dirs_recursively(target_directory, size_limit_kb=5, dry_run=True)

    # 3. 当你确认演习结果无误后，可以将 dry_run 设置为 False 来执行真正的删除
    print("\n" + "=" * 50 + "\n")
    print("第二步：如果演习结果符合预期，请取消下面代码的注释以执行真实删除。")

    anwser = input("你是否希望执行真实删除操作？(yes/no): ")
    if anwser.lower() == "yes":
        delete_small_dirs_recursively(target_directory, size_limit_kb=5, dry_run=False)
    else:
        print("未执行真实删除操作。")
