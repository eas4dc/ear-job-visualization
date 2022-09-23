# pip install mysql-connector
import os
import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


event_types = {0: "energy_policy_new_freq",
               1: "global_energy_policy",
               2: "energy_policy_fails",
               3: "dynais_off",
               4: "earl_state",
               5: "earl_phase",
               6: "policy_phase",
               7: "mpi_load_balance",
               8: "earl_opt_accuracy" }


def db_conn(ear_conf_file):
    db = {}
    for line in open(ear_conf_file):
        li = line.strip()
        if li.startswith("DBIp"):
            db['host'] = li.split("=")[1]
        if li.startswith("DBCommandsUser"):
            db['user'] = li.split("=")[1]
        if li.startswith("DBCommandsPassw"):
            db['password'] = li.split("=")[1]
        if li.startswith("DBDatabase"):
            db['database'] = li.split("=")[1]
        if li.startswith("DBPort"):
            db['port'] = li.split("=")[1]

    # Connect to MariaDB
    conn = mysql.connector.connect( user=db['user'], password=db['password'],
            host=db['host'], database=db['database'],port=db['port'])
    cursor = conn.cursor()

    # TODO test the connector
    return conn, cursor


def query(cursor, job_id, node_id=None):
    if node_id == None:
        event_query = "SELECT job_id, node_id, timestamp as time_stamp, event_type, freq as event_class "\
                      "FROM Events WHERE job_id = %s AND event_type in (4,5,8)"
        cursor.execute(event_query, (job_id,))
    else:
        event_query = "SELECT job_id, node_id, timestamp as time_stamp, event_type, freq as event_clas "\
                      "FROM Events WHERE job_id = %s AND node_id = %s  AND event_type in (4,5,8)"
        cursor.execute(event_query, (job_id, node_id))

    values = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(values, columns=columns)
    df.dropna(inplace=True)

    # Add elapsed time to dataframe
    t0 = df.time_stamp[0]
    elapsed_time = []
    for i in df.index:
        elapsed_time.append(df.time_stamp[i] - t0)
    df['elapsed_time'] = elapsed_time

    return df


def plot_earl_states(df, node, job_id, app_name, show=False):
    # Identifiers with name and colors
    earl_states = {
        0: ["NO_PERIOD","gray"],
        2: ["EVALUATING_LOCAL_SIGNATURE", "orange"],
        1: ["FIRST_ITERATION", "blue"],
        3: ["SIGNATURE_STABLE", "green"],
        4: ["PROJECTION_ERROR", "red"],
        5: ["RECOMPUTING_N", "pink"],
        6: ["SIGNATURE_HAS_CHANGED","brown"],
        8: ["EVALUATING_GLOBAL_SIGNATURE", "purple"]
    }

    # Data for event_type = 4: "earl_state",
    df4 = df.loc[(df.node_id==node) & (df.event_type==4) & (df.event_class != 7)]

    # For start and end time during the application period
    bounds = {key: [[], []] for key in df4.event_class.unique()}
    index = df4.index
    last_event = df4.event_class[index[0]]
    time_start = df4.elapsed_time[index[0]]-1
    time_end = time_start

    for i in index[1:]:
        current_event = df4.event_class[i]
        if current_event != last_event:
            time_end = df4.elapsed_time[i]
            bounds[last_event][0].append(time_start)
            bounds[last_event][1].append(time_end)
            last_event = current_event
            time_start = time_end

    bounds[last_event][0].append(time_start)
    bounds[last_event][1].append(df4.elapsed_time[i]+1)

    # The graphs
    fig, ax = plt.subplots(1, 1, figsize=[16, 1.8])
    lines = []
    labels = []
    for k in bounds:
        label = earl_states[k][0]
        labels.append(label)
        color = earl_states[k][1]
        line = ax.plot(bounds[k], np.zeros_like(bounds[k]).tolist(), lw=20, solid_capstyle='butt', color=color)
        lines.append(line[0])

    leg = ax.legend(lines, labels, loc=1, ncol=3, fontsize=11)
    for ll in leg.get_lines():
        ll.set_linewidth(10)

    ax.set_ylim(-0.25, 0.75)
    ax.set_xlim(df.elapsed_time.to_list()[0], df.elapsed_time.to_list()[-1])
    ax.set_yticks([])
    ax.set_ylabel(node, rotation=0, labelpad=18, fontsize=13)

    fig.tight_layout()
    ax.text(1.5, 0.5,fontsize=13, s=f"EARL_STATE: {app_name} (job-id {job_id})")

    if show:
        plt.show()
    else:
        plt.savefig(fname=f"earl_states_{node}.png", bbox_inches="tight")


def plot_earl_phases(df, node, job_id, app_name, show=False):
    # Identifiers with name and colors
    earl_phases = {
        1: ["APP_COMP_BOUND","green"],
        2: ["APP_MPI_BOUND", "orange"],
        3: ["APP_IO_BOUND", "red"],
        4: ["APP_BUSY_WAITING", "blue"],
        5: ["APP_CPU_GPU", "gray"],
    }

    # Data for event_type = 4: "earl_state",
    df5 = df.loc[(df.node_id==node) & (df.event_type==5)]

    # For start and end time during the application period
    bounds = {key: [[], []] for key in df5.event_class.unique()}
    index = df5.index
    last_event = df5.event_class[index[0]]
    time_start = df5.elapsed_time[index[0]]-1
    time_end = time_start

    for i in index[1:]:
        current_event = df5.event_class[i]
        if current_event != last_event:
            time_end = df5.elapsed_time[i]
            bounds[last_event][0].append(time_start)
            bounds[last_event][1].append(time_end)
            last_event = current_event
            time_start = time_end

    bounds[last_event][0].append(time_start)
    bounds[last_event][1].append(df5.elapsed_time[i]+1)

    # The graphs
    fig, ax = plt.subplots(1, 1, figsize=[16, 1.8])
    lines = []
    labels = []
    for k in bounds:
        label = earl_phases[k][0]
        labels.append(label)
        color = earl_phases[k][1]
        line = ax.plot(bounds[k], np.zeros_like(bounds[k]).tolist(), lw=20, solid_capstyle='butt', color=color)
        lines.append(line[0])

    leg = ax.legend(lines, labels, loc=1, ncol=3, fontsize=11)
    for ll in leg.get_lines():
        ll.set_linewidth(10)

    ax.set_ylim(-0.25, 0.75)
    ax.set_xlim(df.elapsed_time.to_list()[0], df.elapsed_time.to_list()[-1])
    ax.set_yticks([])
    ax.set_ylabel(node, rotation=0, labelpad=18, fontsize=13)

    fig.tight_layout()
    ax.text(1.5, 0.5,fontsize=13, s=f"EARL_PHASE: {app_name} (job-id {job_id})")

    if show:
        plt.show()
    else:
        plt.savefig(fname=f"earl_phase_{node}.png", bbox_inches="tight")


def plot_earl_opt_accuracy(df, node, job_id, app_name, show=False):
    # Identifiers with name and colors
    earl_opt_accuracy = {
        0: ["OPT_NOT_READY","blue"],
        3: ["OPT_TRY_AGAIN", "orange"],
        1: ["OPT_OK", "green"],
        2: ["OPT_NOT_OK", "red"],
    }

    # Data for event_type = 4: "earl_state",
    df8 = df.loc[(df.node_id==node) & (df.event_type==8)]

    # For start and end time during the application period
    bounds = {key: [[], []] for key in df8.event_class.unique()}
    index = df8.index
    last_event = df8.event_class[index[0]]
    time_start = df8.elapsed_time[index[0]]-1
    time_end = time_start

    for i in index[1:]:
        current_event = df8.event_class[i]
        if current_event != last_event:
            time_end = df8.elapsed_time[i]
            bounds[last_event][0].append(time_start)
            bounds[last_event][1].append(time_end)
            last_event = current_event
            time_start = time_end

    bounds[last_event][0].append(time_start)
    bounds[last_event][1].append(df8.elapsed_time[i]+1)

    # The graphs
    fig, ax = plt.subplots(1, 1, figsize=[16, 1.8])
    lines = []
    labels = []
    for k in bounds:
        label = earl_opt_accuracy[k][0]
        labels.append(label)
        color = earl_opt_accuracy[k][1]
        line = ax.plot(bounds[k], np.zeros_like(bounds[k]).tolist(), lw=20, solid_capstyle='butt', color=color)
        lines.append(line[0])

    leg = ax.legend(lines, labels, loc=1, ncol=3, fontsize=11)
    for ll in leg.get_lines():
        ll.set_linewidth(10)

    ax.set_ylim(-0.25, 0.75)
    ax.set_xlim(df.elapsed_time.to_list()[0], df.elapsed_time.to_list()[-1])
    ax.set_yticks([])
    ax.set_ylabel(node, rotation=0, labelpad=18, fontsize=13)

    fig.tight_layout()
    ax.text(1.5, 0.5,fontsize=13, s=f"EARL_OPT_ACCURACY: {app_name} (job-id {job_id})")

    if show:
        plt.show()
    else:
        plt.savefig(fname=f"earl_opt_accuracy_{node}.png", bbox_inches="tight")


def main():
    """ Entry method. """
    # Read ear.conf to get DB connection
    ear_etc_path = os.getenv("EAR_ETC")
    ear_conf_file = os.path.join(ear_etc_path, "ear/ear.conf")

    conn, cursor = db_conn(ear_conf_file)
    job_id = 1536577
    df = query(cursor, job_id)

    list_nodes = df.node_id.unique().tolist()
    for node in list_nodes:
        plot_earl_states(df, node, job_id, "SPO")
        plot_earl_phases(df, node, job_id, "SPO")
        plot_earl_opt_accuracy(df, node, job_id, "SPO")

    # Close DB connection
    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
