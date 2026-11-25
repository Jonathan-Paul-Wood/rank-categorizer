import os

import uvicorn

from .config import Settings


def main():
    os.umask(0o077)
    settings = Settings.from_env()
    uvicorn.run("kev_service.app:create_app", factory=True, host=settings.host, port=settings.port,
                workers=1, access_log=False)


if __name__ == "__main__":
    main()
