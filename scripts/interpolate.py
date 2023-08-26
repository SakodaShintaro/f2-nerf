import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
import numpy as np


def interpolate_pose_in_time(df: pd.DataFrame, target_timestamp_list: pd.Series) -> pd.DataFrame:
    """ dfというデータフレームを、target_timestamp_listが示すタイムスタンプに合わせて補間する
    制約)
    * dfとtarget_timestamp_listはタイムスタンプでソートされていること
    * dfはtarget_timestamp_listを覆うような前後に広い区間のタイムスタンプを持つこと
    出力)
    * dfと同じカラムを持ち、長さがtarget_timestamp_listと同じであるデータフレーム
    """
    POSITIONS_KEY = ['x', 'y', 'z']
    ORIENTATIONS_KEY = ['qw', 'qx', 'qy', 'qz']
    result_df = pd.DataFrame(columns=df.columns)
    target_index = 0
    df_index = 0
    while df_index < len(df) - 1 and target_index < len(target_timestamp_list):
        curr_time = df.iloc[df_index]['timestamp']
        next_time = df.iloc[df_index + 1]['timestamp']
        target_time = target_timestamp_list[target_index]

        # target_timeを挟み込むようなdf_indexを探す
        if not (curr_time <= target_time <= next_time):
            df_index += 1
            continue

        curr_weight = (next_time - target_time) / (next_time - curr_time)
        next_weight = 1.0 - curr_weight

        curr_position = df.iloc[df_index][POSITIONS_KEY]
        next_position = df.iloc[df_index + 1][POSITIONS_KEY]
        target_position = curr_position * curr_weight + next_position * next_weight

        curr_orientation = df.iloc[df_index][ORIENTATIONS_KEY]
        next_orientation = df.iloc[df_index + 1][ORIENTATIONS_KEY]
        curr_r = Rotation.from_quat(curr_orientation)
        next_r = Rotation.from_quat(next_orientation)
        slerp = Slerp([curr_time, next_time],
                      Rotation.concatenate([curr_r, next_r]))
        target_orientation = slerp([target_time]).as_quat()[0]

        target_row = df.iloc[df_index].copy()
        target_row['timestamp'] = target_timestamp_list[target_index]
        target_row[POSITIONS_KEY] = target_position
        target_row[ORIENTATIONS_KEY] = target_orientation
        result_df = result_df.append(target_row)
        target_index += 1
    return result_df


if __name__ == "__main__":
    df = pd.DataFrame(columns=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    # 秒数0で(x, y, z) = (0, 0, 0), 向きX軸方向
    default_rot = Rotation.from_quat(np.array([0, 0, 0, 1]))
    quat1 = default_rot.as_quat()
    df.loc[0] = [0, 0, 0, 0, quat1[0], quat1[1], quat1[2], quat1[3]]

    # 秒数1で(x, y, z) = (0, 0, 0), yaw 90度回転
    r2 = Rotation.from_rotvec(np.array([0, 0, 1]) * (90 * (np.pi / 180)))
    quat2 = (r2 * default_rot).as_quat()
    df.loc[1] = [1, 0, 0, 0, quat2[0], quat2[1], quat2[2], quat2[3]]

    target_timestamp_list = pd.Series([0.5])

    # 関数をテスト
    result_df = interpolate_pose_in_time(df, target_timestamp_list)

    # 出力結果の表示
    print(result_df)
    r_ans = Rotation.from_rotvec(np.array([0, 0, 1]) * (45 * (np.pi / 180)))
    print(r_ans.as_quat())
