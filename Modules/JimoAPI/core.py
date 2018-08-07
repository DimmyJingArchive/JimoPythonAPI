import numpy as np
import subprocess
import socket
import struct

import JimoAPI.connector

default_port = 3300

pack_map = {'bool': ('?', 1), 'i8': ('b', 1), 'u8': ('B', 1), 'i16': ('h', 2), 'u16': ('H', 2),
            'i32': ('i', 4), 'u32': ('I', 4), 'd32': ('f', 4), 'i64': ('q', 8), 'u64': ('Q', 8), 'd64': ('d', 8)}
type_map = {'string': 'str', 'double': 'd64', 'u64': 'u64', 'int64': 'i64', 'datetime': 'str',
            'time': 'str', 'timestamp': 'i64', 'date': 'str', 'number': 'd64', 'bool': 'bool'}


def pack_data(string, data):
    if type(data) is str:
        string += struct.pack('<i', len(data)) + data.encode('ascii')
    else:
        string += struct.pack('<' + pack_map[data[1]][0], data[0])


def unpack_data(string, idx, data_type, length=None, byte=True):
    if length is not None:
        temp = string[idx:idx+length]
        idx += length
        return temp, idx
    elif data_type == 'str':
        temp = string[idx+4:idx+4+struct.unpack('<i', string[idx:idx+4])[0]]
        idx += struct.unpack('<i', string[idx:idx+4])[0] + 4
        return temp, idx
    elif data_type == 'd64' and byte:
        idx += 9
        return struct.unpack('<d', string[idx-8:idx])[0], idx
    else:
        idx += pack_map[data_type][1]
        return struct.unpack('<'+pack_map[data_type][0], string[idx-pack_map[data_type][1]:idx])[0], idx


def first_not_none(array, extra_flags=''):
    last_table = len(array)-1
    for idx, i in enumerate(array):
        if i is not None:
            try:
                if not getattr(i, extra_flags):
                    return (i, idx, True)
            except AttributeError:
                return (i, idx, True)
            last_table = idx
    return (array[last_table], last_table, False)


def decode_bytes(array):
    try:
        return array.decode('ascii')
    except UnicodeError:
        try:
            return array.decode('utf8')
        except UnicodeError:
            try:
                return array.decode('gb2312')
            except UnicodeError as e:
                print("unable to decode bytes, error: {}".format(repr(e)))
                return array


def get_cache(operation):
    raise NotImplementedError


class JimoTable():
    def serialize_from(self, data, idx):
        self.struct_type, idx = unpack_data(data, idx, 'str')
        self.cost, idx = unpack_data(data, idx, 'd64', byte=False)
        self.count, idx = unpack_data(data, idx, 'i64')
        self.pagesize, idx = unpack_data(data, idx, 'i64')
        self.msg, idx = unpack_data(data, idx, 'str')
        self.msg = decode_bytes(self.msg)
        if self.struct_type == b'msg':
            self.msg_id, idx = unpack_data(data, idx, 'u32')
            if self.msg_id > 100000:
                self.error_msg, idx = unpack_data(data, idx, 'str')
                self.error_msg = decode_bytes(self.error_msg)
                if self.msg_id == 100307:
                    raise JimoAPI.connector.ProgrammingError(self.error_msg)
                raise JimoAPI.connector.OperationalError(self.error_msg)
        else:
            self.is_empty = False
            self.num_columns, idx = unpack_data(data, idx, 'u32')
            self.columns = {}
            self.fields = []
            for _ in range(self.num_columns):
                column = JimoColumn()
                idx = column.serialize_from(data, idx)
                self.columns[column.field] = column
                self.fields.append(column.field)
            length, idx = unpack_data(data, idx, 'i64')
            crt_idx = idx
            self.data, idx = unpack_data(data, idx, 'str', length)
            self.size = 0
            while crt_idx < idx:
                self.size += 1
                for column_name in self.fields:
                    is_null, crt_idx = unpack_data(data, crt_idx, 'u8')
                    if is_null:
                        self.columns[column_name].data.append('NULL')
                    else:
                        temp, crt_idx = unpack_data(data, crt_idx, self.columns[column_name].type)
                        if type(temp) == bytes or type(temp) == bytearray:
                            temp = decode_bytes(temp)
                        self.columns[column_name].data.append(temp)
            for i, j in self.columns.items():
                self.columns[i].data = np.array(j.data)
            idx = crt_idx
            self.idx = 0
        return idx

    def get_row(self):
        if self.idx == self.size:
            return None
        row = {i: j.data[self.idx] for i, j in self.columns.items()}
        self.idx += 1
        return row

    def get_num_rows(self, size=1):
        if self.idx == self.size:
            return None, True
        temp = {i: j.data[self.idx:self.idx+size] for i, j in self.columns.items()}
        self.idx = min(self.idx + size, self.size)
        if self.idx == self.size:
            return temp, True
        return temp, False

    def get_all_rows(self):
        if self.idx == self.size:
            return None
        temp = {i: j.data[self.idx:] for i, j in self.columns.items()}
        self.idx = self.size
        return temp

    def get_column(self, name):
        return self.columns[name].data

    def __repr__(self):
        if self.struct_type == b'msg':
            return 'Msg: "{}"'.format(self.msg)
        else:
            return 'Msg: "{}", Fields: {}'.format(self.msg, self.columns)


class JimoColumn():
    def serialize_from(self, data, idx):
        self.field, idx = unpack_data(data, idx, 'str')
        self.field = decode_bytes(self.field)
        self.alias, idx = unpack_data(data, idx, 'str')
        self.alias = decode_bytes(self.alias)
        self.format, idx = unpack_data(data, idx, 'str')
        self.type, idx = unpack_data(data, idx, 'str')
        self.type = type_map[decode_bytes(self.type)]
        self.data = []
        return idx

    def __repr__(self):
        return '"name:{},type:{},data:{}"'.format(self.field, self.type, self.data)


class JimoDB():
    def connect(self, user, password, host, port, db):
        # Connect
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10)
        try:
            self.socket.connect((host, port))
        except (ConnectionRefusedError, socket.timeout):
            print('Connection Refused, host: {}, port: {}'.format(host, port))
            self.connect_flag = False
            return False
        self.connect_flag = True
        # Login
        res = bytearray()
        pack_data(res, 'login')
        pack_data(res, user)
        pack_data(res, password)
        pack_data(res, 'zky_sysuser')
        # Send and Receive Data
        body = self.send_and_recv(res)
        if body is None:
            return False
        idx = 1
        self.build_user, idx = unpack_data(body, idx, 'str')
        self.build_date, idx = unpack_data(body, idx, 'str')
        self.build_version, idx = unpack_data(body, idx, 'str')
        self.build_number, idx = unpack_data(body, idx, 'i32')
        self.urldocid, idx = unpack_data(body, idx, 'u64')
        self.ssid, idx = unpack_data(body, idx, 'str')
        # Run JQL
        cmd = 'use {};'.format(db)
        self.run_jql(cmd)
        return True

    def run_jql(self, cmd):
        if self.connect_flag is False:
            raise ConnectionRefusedError
        res = bytearray()
        pack_data(res, 'runjql')
        pack_data(res, decode_bytes(self.ssid))
        pack_data(res, (self.urldocid, 'u64'))
        pack_data(res, cmd)
        pack_data(res, (1, 'u8'))
        pack_data(res, (0, 'u8'))
        body = self.send_and_recv(res)
        if body is None:
            return
        idx = 1
        cost, idx = unpack_data(body, idx, 'd64', byte=False)
        size, idx = unpack_data(body, idx, 'u32')
        # NOTE: multiple tables as a result is not supported since no query is expected to return two or more tables
        table = JimoTable()
        table.serialize_from(body, idx)
        return table

    def close(self):
        self.socket.close()

    def send_and_recv(self, data):
        self.socket.send(struct.pack('>i', len(data)) + data)
        length = struct.unpack('>i', self.socket.recv(4))[0]
        body = bytearray()
        while len(body) < length:
            body += self.socket.recv(4096)
        if not body[0]:
            print(decode_bytes(body[5:]))
            return None
        return body


class JimoLoadingBar():
    prefix = '|'
    postfix = '|{processed:5} /{pending:5}, {percentage:7.2%}'
    phases = ' ▏▎▍▌▋▊▉█'

    def __init__(self, pending=None):
        self.rows, self.columns = map(lambda i: int(i), subprocess.check_output(['stty', 'size']).split())
        self.data = {'percentage': 0., 'processed': 0, 'pending': pending}
        self.pending = pending

    def update(self, processed, prefix_data={None: None}, postfix_data={None: None}):
        if self.pending is not None:
            self.data.update({'processed': processed})
        self.data['percentage'] = min(processed / (self.pending or 1.), 1.)
        bar_length = self.columns - len(self.prefix.format(**self.data, **prefix_data)) - len(self.postfix.format(**self.data, **postfix_data)) - 1
        occupied_space = self.data['percentage'] * bar_length
        full_element = self.phases[-1] * int(occupied_space)
        partial_element = self.phases[int(occupied_space % 1 * len(self.phases))] if len(full_element) < bar_length else ''
        empty_element = self.phases[0] * (bar_length - len(full_element) - len(partial_element))
        print('\r' + self.prefix.format(**self.data, **prefix_data) + full_element + partial_element +
              empty_element + self.postfix.format(**self.data, **postfix_data), end='', flush=True)
