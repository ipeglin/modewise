fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(modewise_cli::main(std::env::args_os()))
}
