module.exports = {
    apps: [{
        name: "novelfactory-api",
        script: "uvicorn",
        args: "novelfactory.server.app:app --host 0.0.0.0 --port 8000 --log-level info",
        interpreter: "python3",
        instances: 1,
        exec_mode: "fork",
        watch: false,
        max_memory_restart: "4G",
        kill_timeout: 15000,
        listen_timeout: 10000,
        shutdown_with_message: true,
        env: {
            PYTHONPATH: "/app/src",
            PYTHONUNBUFFERED: "1",
        },
        error_file: "/data/logs/pm2-error.log",
        out_file: "/data/logs/pm2-out.log",
        merge_logs: true,
        log_date_format: "YYYY-MM-DD HH:mm:ss Z",
        max_restarts: 5,
        restart_delay: 10000,
        autorestart: true,
    }]
};
