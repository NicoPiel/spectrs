use std::sync::atomic::AtomicBool;

use tracing::{error, info};

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    let mut exit = AtomicBool::new(false);

    Ok(())
}
