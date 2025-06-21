# task_runner.py (高拡張性・並列実行・Git連携・動的タスク生成・自己更新機能付き)
import time
import subprocess
import datetime
import os
import sys

# --- 設定項目 ---

# 実行したいコマンドのテンプレートを定義します。
# {START_PERCENT} と {END_PERCENT} という文字列は、実行時に動的に計算された範囲に置き換えられます。
COMMAND_TEMPLATE = "claude --model opus --dangerously-skip-permissions -p \"you need to be more wiser and do ultrathink to earn more profit, search web rarely high-end trade symbols and install. when you search web, use multi agents with ultrathink.\" --output-format text --verbose"

# 並列で実行するインスタンス（プロセス）の数
NUM_INSTANCES = 1

# Gitリポジトリのローカルパス
# git pull/push を実行したいリポジトリのパスを指定してください。
# 例: "/home/user/my-project/"
GIT_REPO_PATH = "/home/takumi/MT5Claude_Main/"

# 実行間隔（時間） - 次の実行時刻を計算する際の基準となります（現在の実行時+INTERVAL_HOURS : 01に再試行します）
INTERVAL_HOURS = 1

# スクリプトファイルと同じ場所を基準とします
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # __file__ を絶対パスに

# 結果を出力するディレクトリ
OUTPUT_DIR = os.path.join(BASE_DIR, "exec_log")

### 変更・追加点 ###
### 変更・追加ここまで ###


# --- ヘルパー関数 ---

def execute_sync_command(command: str, working_dir: str = None) -> str:
    """
    指定されたコマンドを同期待ちで実行し、標準出力と標準エラーを結合した文字列を返します。
    作業ディレクトリを指定できます。
    """
    print(f"実行中 (同期 in {working_dir or 'current dir'}): {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False,
            cwd=working_dir  # コマンドを実行するディレクトリを指定
        )
        output = f"--- Standard Output ---\n{result.stdout}\n"
        if result.stderr:
            output += f"\n--- Standard Error ---\n{result.stderr}"
        return output
    except Exception as e:
        return f"コマンドの実行中に予期せぬエラーが発生しました: {e}"


def wait_for_next_run(interval_hours: int):
    """
    次の実行時刻まで待機する関数。
    次の実行は、現在から `interval_hours` 時間が経過した後の、
    最初の「XX時01分」に行われる。
    """
    now = datetime.datetime.now()
    
    # 待機後の基準となる時刻を計算 (現在時刻 + 指定時間)
    # 例: nowが13:20でintervalが5時間なら、基準は18:20
    base_time = now + datetime.timedelta(hours=interval_hours)
    
    # 次に実行すべき時刻を計算する
    # 分(minute)を1に設定する
    # 例: 基準が18:20 -> 19:01 に設定
    next_run_time = base_time.replace(minute=1, second=0, microsecond=0)
    
    # 待機時間を計算
    wait_seconds = (next_run_time - now).total_seconds()
    
    if wait_seconds > 0:
        print(f"\n次の実行時刻: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        # 見やすい形式で待機時間を表示
        wait_duration = datetime.timedelta(seconds=int(wait_seconds))
        print(f"待機します... (待機時間: {wait_duration})")
        time.sleep(wait_seconds)

### 変更・追加点 ###
def cleanup_old_files(target_dir: str, days_threshold: int):
    """
    指定されたディレクトリ内の古いファイルを削除する。

    Args:
        target_dir (str): クリーンアップ対象のディレクトリのパス。
        days_threshold (int): この日数以上経過したファイルを削除する。
    """
    print(f"\n--- 古いファイルのクリーンアップを開始します (対象: '{os.path.basename(target_dir)}', 基準: {days_threshold}日以上前) ---")
    if not os.path.isdir(target_dir):
        print(f"警告: クリーンアップ対象ディレクトリ '{target_dir}' が見つかりません。スキップします。")
        print("--- クリーンアップ完了 ---")
        return

    now = datetime.datetime.now()
    threshold_time = now - datetime.timedelta(days=days_threshold)
    
    try:
        file_count = 0
        deleted_count = 0
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            
            if os.path.isfile(file_path):
                file_count += 1
                try:
                    file_mtime_ts = os.path.getmtime(file_path)
                    file_mtime_dt = datetime.datetime.fromtimestamp(file_mtime_ts)
                    
                    if file_mtime_dt < threshold_time:
                        print(f"  削除: {filename} (最終更新: {file_mtime_dt.strftime('%Y-%m-%d %H:%M')})")
                        os.remove(file_path)
                        deleted_count += 1
                except OSError as e:
                    print(f"エラー: ファイル '{file_path}' の削除に失敗しました。: {e}")
                except Exception as e:
                    print(f"エラー: ファイル '{file_path}' の処理中に予期せぬエラーが発生しました。: {e}")
        
        if file_count == 0:
            print("対象ディレクトリにファイルはありませんでした。")
        else:
            print(f"{deleted_count} / {file_count} 個のファイルを削除しました。")

    except Exception as e:
        print(f"エラー: ディレクトリ '{target_dir}' のスキャン中にエラーが発生しました。: {e}")
    
    print("--- クリーンアップ完了 ---")
### 変更・追加ここまで ###


# --- スクリプト本体 ---

def main():
    """
    メインの処理ループ
    """
    # 出力ディレクトリが存在しない場合は作成する
    if not os.path.isdir(OUTPUT_DIR):
        print(f"出力ディレクトリ '{OUTPUT_DIR}' が存在しないため、作成します。")
        try:
            os.makedirs(OUTPUT_DIR)
        except OSError as e:
            print(f"エラー: 出力ディレクトリの作成に失敗しました。: {e}")
            return


    print("スクリプトを開始します。（動的タスク生成・並列実行・自己更新モード）")
    print(f"並列インスタンス数: {NUM_INSTANCES}件")
    print(f"実行間隔の基準: {INTERVAL_HOURS}時間")
    print(f"Gitリポジトリ: {GIT_REPO_PATH}")
    print(f"出力先: {OUTPUT_DIR}")
    print("中断するには Ctrl+C を押してください。")

    while True:
        try:
            ### 変更・追加点 ###
            # 0. 古いファイルをクリーンアップする
            cleanup_old_files(os.path.join(GIT_REPO_PATH, "exec_log"), 1)

            # 2. 実行するコマンドリストを動的に生成
            commands_to_run = []
            for _ in range(NUM_INSTANCES):
               
                command = COMMAND_TEMPLATE
                commands_to_run.append(command)


            # 実行前の準備
            timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
            filename = os.path.join(OUTPUT_DIR, f"{timestamp}.txt")
            output_content = ""

            # 3. git pull と自己更新チェック
            print("\ngit pullから開始します。")
            pull_output = execute_sync_command("git pull", working_dir=GIT_REPO_PATH)

            if "Already up to date" not in pull_output:
                print(f"!!! リポジトリが更新されました。自動的に再起動します。 !!!")
                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"実行日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"スクリプト更新のため、新しいプロセスで再起動します。\n\n")
                        f.write("********** git pull **********\n")
                        f.write(pull_output + "\n")
                except IOError as e:
                    print(f"警告: 再起動前のログ書き込みに失敗しました: {e}")
                
                os.execv(sys.executable, ['python3'] + sys.argv)
            
            # ループを継続する場合の処理
            if not commands_to_run:
                #: 待機関数を呼び出す
                wait_for_next_run(INTERVAL_HOURS)
                continue # ループの先頭に戻る

            output_content += "********** git pull **********\n"
            output_content += pull_output + "\n\n"
            
            # 4. 生成されたコマンドを並列実行
            processes = []
            print(f"{len(commands_to_run)}個のタスクを並列で実行します。")
            for i, command in enumerate(commands_to_run):
                print(f"  タスク{i+1}を開始: {command}")
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    cwd=GIT_REPO_PATH
                )
                processes.append((proc, command))

            # 5. すべてのタスクの完了を待つ
            print("すべてのタスクの完了を待っています...")
            claude_outputs_content = ""
            for i, (proc, command) in enumerate(processes):
                stdout, stderr = proc.communicate()
                print(f"  タスク{i+1}が完了しました。")

                claude_outputs_content += f"********** execute command (task {i+1}) **********\n"
                claude_outputs_content += f"Command: {command}\n"
                claude_outputs_content += f"--- Standard Output ---\n{stdout}\n"
                if stderr:
                    claude_outputs_content += f"\n--- Standard Error ---\n{stderr}"
                claude_outputs_content += "\n\n"
            
            output_content += claude_outputs_content
            output_content += "\n\n\n\n\n\n\n\n"

            # 6. コマンド実行結果をファイルに書き込み
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"実行日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(output_content)
                print(f"結果を {filename} に出力しました。")
            except IOError as e:
                print(f"エラー: ファイル '{filename}' に書き込めませんでした。: {e}")

            # 8. git add, commit & push
            print("すべてのタスクが完了したため、git add, commit, pushを実行します。")
            execute_sync_command("git add .", working_dir=GIT_REPO_PATH)
            execute_sync_command('git commit -m "Scheduled task result"', working_dir=GIT_REPO_PATH)
            execute_sync_command("git push", working_dir=GIT_REPO_PATH)
            
            # 9. 次の実行まで待機
            wait_for_next_run(INTERVAL_HOURS)

        except KeyboardInterrupt:
            print("\nスクリプトを終了します。")
            break
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e}")
            print("10分後に処理を再試行します...")
            time.sleep(600)

if __name__ == "__main__":
    main()
