import time

import JimoAPI.core
import JimoAPI.cursor

default_port = 3300


def connect(*args, **kwargs):
    return JimoConnection(*args, **kwargs)


class JimoConnection():
    def __init__(self, user='', password='', host='127.0.0.1', port=default_port, database='db'):
        if user == '' or password == '':
            print("Please enter a username and password")
            return
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.db = JimoAPI.core.JimoDB()
        self.db.connect(user, password, host, port, database)
        self.main_cursor = self.cursor()

    def close(self):
        self.db.close()

    def commit(self):
        return self.main_cursor.execute('COMMIT;')

    def config(self, **kwargs):
        for i, j in kwargs.items():
            setattr(self, i, j)

    def connect(self, **kwargs):
        self.config(**kwargs)
        self.reconnect()

    def cursor(self):
        # TODO: follow mysql implementation
        return JimoAPI.cursor.JimoCursor(self)

    def cmd_change_user(self, username='', password='', database='', charset=33):
        self.user = username or self.user
        self.password = password or self.password
        self.database = database or self.database
        if not hasattr(self, 'charset'):
            self.charset = charset
        elif charset != 33:
            self.charset = charset

    def cmd_debug(self):
        raise NotSupportedError("debug not supported yet")

    def cmd_init_db(self, database):
        return self.main_cursor.execute('USE {};'.format(database))

    def cmd_ping(self):
        return self.main_cursor.execute('PING;')

    def cmd_query(self, statement):
        return self.main_cursor.execute(statement)

    def cmd_query_iter(self, cmd):
        return self.main_cursor.execute(cmd, multi=True)

    def cmd_quit(self):
        self.close()

    def cmd_refresh(self, options):
        raise NotSupportedError("cmd_refresh not supported yet")

    def cmd_reset_connection(self):
        self.close()
        self.reconnect()

    def cmd_shutdown(self):
        self.close()
        return self.main_cursor.execute("SHUTDOWN SYSTEM;")

    def cmd_statistics(self):
        raise NotSupportedError("statistics not supported yet")

    def disconnect(self):
        self.close()
        return self.main_cursor.execute("QUIT;")

    def get_row(self):
        return self.main_cursor.fetchone()

    def get_rows(self, count=None):
        if count is not None:
            return self.main_cursor.fetchmany(count)
        if count is None:
            return self.main_cursor.fetchall()

    def get_server_info(self):
        raise NotSupportedError("get_server_info not supported yet")

    def get_server_version(self):
        raise NotSupportedError("get_server_version not supported yet")

    def is_connected(self):
        try:
            self.db.run_jql("TEST;")
            return True
        except ConnectionRefusedError:
            return False

    def isset_client_flag(self):
        return hasattr(self, 'client_flag')

    def ping(self):
        return self.cmd_ping()

    def reconnect(self, attempts=1, delay=0):
        for _ in range(attempts):
            result = self.db.connect(self.user, self.password, self.host, self.port, self.database)
            if result:
                break
            time.sleep(delay)
        return result

    def reset_session(self, user_variables, session_variables):
        raise NotSupportedError("reset_session not supported yet")

    def rollback(self):
        raise NotSupportedError("rollback not supported yet")

    def set_charset_collation(self, charset=None, collation=None):
        if not hasattr(self, 'charset'):
            self.charset = charset
        elif charset is not None:
            self.charset = charset
        if not hasattr(self, 'collation'):
            self.collation = collation
        elif collation is not None:
            self.collation = collation

    def set_client_flag(self, flags):
        self.client_flag = flags

    def shutdown(self):
        self.close()

    def start_transaction(self, consistent_snapshot, isolation_level, readonly):
        raise NotSupportedError("start_transaction not supported yet")

    @property
    def unread_results(self):
        return self.main_cursor.result.is_empty


class JimoConnectionPool():
    def __init__(self, pool_name=None, pool_size=5, pool_reset_session=True, user='', password='', host='127.0.0.1', port=default_port, database='db'):
        self.pool_name = pool_name
        self.pool_size = pool_size
        self.pool_reset_session = pool_reset_session
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.connections = [None]*pool_size

    def add_connection(self, cnx=None):
        if cnx is None:
            cnx = JimoConnection(self.user, self.password, self.host, self.port, self.database)
        spot = None
        for idx, i in enumerate(self.connections):
            if i is None:
                spot = idx
                break
        if spot is None:
            raise PoolError("Not enough spaces for connections")
        self.connections[spot] = PooledJimoConnection(self, cnx)

    def get_connection(self):
        temp, spot, flag = JimoAPI.core.first_not_none(self.connections, 'is_occupied')
        if not flag:
            raise PoolError("Not enough connections")
        self.connections[spot].is_occupied = True
        return self.connections[spot]

    def fill(self):
        for i in self.connections:
            if i is None:
                self.add_connection()

    def set_config(self, **kwargs):
        for i, j in kwargs.items():
            setattr(self, i, j)


class PooledJimoConnection(JimoConnection):
    def __init__(self, cnxpool, cnx):
        self.cnxpoolname = cnxpool.pool_name
        self.__dict__.update(cnx.__dict__)
        self.is_occupied = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.is_occupied = False

    def config(self, **kwargs):
        raise PoolError("Cannot config pooled connection, config the connection pool instead")

    @property
    def pool_name(self):
        return self.cnxpoolname


class _errorcode(dict):
    __getattr__ = dict.get


errorcode = _errorcode({
    'DATABASE_ERROR': 1,
    'DATA_ERROR': 2,
    'INTEGRITY_ERROR': 3,
    'INTERNAL_ERROR': 4,
    'NOT_SUPPORTED_ERROR': 5,
    'OPERATIONAL_ERROR': 6,
    'PROGRAMMING_ERROR': 7,
    'INTERFACE_ERROR': 8,
    'POOL_ERROR': 9})


class Error(Exception):
    def __init__(self, msg='', errno=None):
        self.msg = msg
        self.errno = errno

    def __repr__(self):
        return self.msg


class DatabaseError(Error):
    errno = errorcode.DATABASE_ERROR


class DataError(DatabaseError):
    errno = errorcode.DATA_ERROR


class IntegrityError(DatabaseError):
    errno = errorcode.INTEGRITY_ERROR


class InternalError(DatabaseError):
    errno = errorcode.INTERNAL_ERROR


class NotSupportedError(DatabaseError):
    errno = errorcode.NOT_SUPPORTED_ERROR


class OperationalError(DatabaseError):
    errno = errorcode.OPERATIONAL_ERROR


class ProgrammingError(DatabaseError):
    errno = errorcode.PROGRAMMING_ERROR


class InterfaceError(Error):
    errno = errorcode.INTERFACE_ERROR


class PoolError(Error):
    errno = errorcode.POOL_ERROR


class Warning(Exception):
    pass
