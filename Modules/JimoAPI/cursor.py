import regex

import JimoAPI.core
import JimoAPI.connector


class JimoCursor():
    def __init__(self, connection=None):
        self.connection = connection
        self.result = None
        self.current_table = None

    def callproc(self):
        raise JimoAPI.connector.NotSupportedError("callproc not supported yet")

    def close(self):
        self.connection = None
        self.result = None
        self.current_table = None

    def execute(self, operation, params=None, multi=False):
        if multi:
            cmd_list = regex.split(r"\([^)]*\)(*SKIP)(*F)|;", operation.strip(';'))

            def iterate():
                for command in cmd_list:
                    self.execute(command+';')
                    yield self
            return iterate()
        if params is not None:
            try:
                operation = operation % params
            except TypeError:
                if type(params) in [tuple, list]:
                    operation = operation.format(*params)
                else:
                    operation = operation.format(params)
        try:
            self.result = self.connection.db.run_jql(operation)
        except ConnectionRefusedError:
            print("Database not connected, using cache")
            try:
                self.result = JimoAPI.core.get_cache(operation)
            except KeyError:
                print("Cache does not exist, skipping operation {}".format(operation))
                return
        self.last_statement = operation
        self.msg = self.result.msg
        return self.msg

    def executemany(self, operation, params):
        try:
            operation % params[0]
            fmt = False
        except TypeError:
            fmt = True
        try:
            for param in params:
                if fmt:
                    self.connection.db.run_jql(operation % param)
                elif type(param) in [tuple, list]:
                    self.connection.db.run_jql(operation.format(*param))
                else:
                    self.connection.db.run_jql(operation.format(param))
        except ConnectionRefusedError:
            print("Database not connected, using cache")
            try:
                self.result = JimoAPI.core.get_cache(operation)
            except KeyError:
                print("Cache does not exist, skipping operation {}".format(operation))
                return
        self.last_statement = operation

    def fetchall(self):
        if not self.result.is_empty:
            self.result.is_empty = True
            return self.result.get_all_rows()
        # TODO raise error

    def fetchmany(self, size=1):
        if not self.result.is_empty:
            rows, is_empty = self.result.get_num_rows(size)
            if is_empty:
                self.result.is_empty = True
            return rows
        # TODO raise error

    def fetchone(self, dictionary=False):
        if not self.result.is_empty:
            row = self.result.get_row()
            if row is None:
                self.result.is_empty = True
            return row
        # TODO raise error

    def fetchwarnings(self):
        if self.result is not None:
            if hasattr(self.result, 'error_msg'):
                return self.result.error_msg
            return self.result.msg
        # TODO raise error

    @property
    def column_names(self):
        if self.result is not None:
            return self.result.fields
        # TODO raise error

    @property
    def column_alias(self):
        if self.result is not None:
            return [self.result.columns[i].alias for i in self.result.columns]
        # TODO raise error

    @property
    def description(self):
        raise JimoAPI.connector.NotSupportedError("description property not supported yet")

    @property
    def lastrowid(self):
        raise JimoAPI.connector.NotSupportedError("lastrowid property not supported yet")

    @property
    def rowcount(self):
        if self.result is not None and not self.result.is_empty:
            return self.result.size - self.result.idx
        return 0

    @property
    def with_rows(self):
        if self.result is not None:
            return self.result.is_empty
        # TODO: raise error

    @property
    def statement(self):
        return self.last_statement if hasattr(self, 'last_statement') else None
