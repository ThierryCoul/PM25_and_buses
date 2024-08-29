library('raster')
library('sp')
library('sf')
library('ggplot2')
library('dplyr')
library('tidyverse')
library("rnaturalearth")

# Importing the shapefile
setwd('/Users/thierrycoulibaly/Library/CloudStorage/OneDrive-KyushuUniversity/Work/Jilan')
ASEAN_shapefile <- 'Shapefiles/5_ASEAN_countries_GID_1.shp'
Provinces_ouline_shp <- st_read(ASEAN_shapefile)
Provinces_ouline_shp <- st_make_valid(Provinces_ouline_shp)
Provinces_ouline_shp_simpl <- st_simplify(Provinces_ouline_shp, preserveTopology = TRUE, dTolerance = 1000)

# Importing raster data of pollution
Pollution_2020_tif <- raster('Shapefiles/PM25_2020.tif')
Extent <- extent(90, 150, -10, 20)
Pollution_2020_tif <- crop(Pollution_2020_tif, Extent)

# Convert raster to data frame for ggplot2
df <- as.data.frame(Pollution_2020_tif, xy=TRUE)
df_2 <- drop_na(df)

world <- ne_countries(scale = "medium", returnclass = "sf")

# Ploting
ggplot(data = Provinces_ouline_shp_simpl) +
  geom_sf(data = world, fill='grey90') +
  geom_sf(aes(fill = FIRST_NA_1), color = 'grey70') +
  theme_void() +
  xlim(90, 150) +  # Set your desired x limits
  ylim(-10, 25) +  # Set your desired y limits
  labs(fill = "Country")

# Exporting the plot
p0 <- last_plot()
ggsave("Unit of area.png", p0, width = 8, height = 8, 
       dpi = 600, type = "cairo-png", bg = "white")

ggplot() +
  geom_sf(data = world, fill='grey90') +
  geom_raster(data = df_2, aes(x = x, y = y, fill = PM25_2020)) +
  #scale_fill_viridis_c() +
  scale_fill_gradient(low = 'purple',high = 'yellow')+
  #coord_fixed() +  
  geom_sf(data = Provinces_ouline_shp_simpl_2, fill = NA, color = 'grey90') +
  xlim(90, 150) +  # Set your desired x limits
  ylim(-10, 20) +  # Set your desired y limits
  theme_void() +
  labs(fill = "PM 2.5")

# Exporting the plot
p0 <- last_plot()
ggsave("PM25.png", p0, width = 8, height = 8, 
       dpi = 600, type = "cairo-png", bg = "white")

# Exporting the plot
p0 <- last_plot()
ggsave("Unit of area.png", p0, width = 8, height = 8, 
       dpi = 600, type = "cairo-png", bg = "white")
