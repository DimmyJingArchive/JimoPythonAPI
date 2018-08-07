import JimoAPI.connector

main_cnx_pool = JimoAPI.connector.JimoConnectionPool('main_pool', 10, True, 'root', '12345', '192.168.99.230')
main_cnx_pool.fill()


class TempCursor():
    def __enter__(self):
        self.cnx = main_cnx_pool.get_connection()
        self.cursor = self.cnx.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_value, traceback):
        self.cnx.close()
        self.cursor.close()
