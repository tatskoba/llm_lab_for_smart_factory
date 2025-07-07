CREATE TABLE `demo_db`.`time_series_data` (
    `ml_type` VARCHAR(32) NOT NULL , 
    `judge` VARCHAR(32) NOT NULL ,
    `class` VARCHAR(32) NOT NULL ,
    `description` TEXT NULL , 
    `date_time` DATETIME NULL , 
    `serial` VARCHAR(32) NOT NULL ,
    `stage` VARCHAR(32) NULL , 
    `parameter` VARCHAR(32) NULL , 
    `csv_file` VARCHAR(32) NOT NULL , 
    `image_file` VARCHAR(32) NOT NULL,
    PRIMARY KEY (`csv_file`)
) ENGINE = InnoDB;