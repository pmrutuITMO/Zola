import os
import re
from pyspark.sql import SparkSession, Row, window, functions as ssql_f
import configparser as cp
import weasyprint as wp

# Get AWS IAM credentials for S3 access.
config = cp.ConfigParser()
config.read(os.path.expanduser("~/.aws/credentials"))
aws_profile = 'default'
access_id = config.get(aws_profile, "aws_access_key_id") 
access_key = config.get(aws_profile, "aws_secret_access_key")

# Obtain a Spark session for our application run then set log level.
spark = SparkSession.builder.appName("PJM Zola Electric 2-Gram Challenge").getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Configure AWS S3 access for Spark using S3A interface
hadoop_conf=spark.sparkContext._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
hadoop_conf.set("fs.s3a.access.key", access_id)
hadoop_conf.set("fs.s3a.secret.key", access_key)

# Get LZO sequence file using S3A interface 
lines = spark.sparkContext.sequenceFile("s3a://datasets.elasticmapreduce/ngrams/books/20090715/eng-1M/2gram/data")

# Mapping function to convert the data from sequence file to row objects
# Convert years to decades for easier processing later.
# Trim strings and lower the case to avoid case sensitive issues later.
# Split each 2-gram into 2 columns; each column for each word in the 2-gram.
def dataConverter(line):
	fields = line[1].split("\t")
	twogram = str(fields[0]).strip().lower().split(" ")
	first_column = twogram[0].strip()
	last_column = "" # to account for terms proceeded by empty space.
	
	if len(twogram) == 2:
		last_column = twogram[1].strip()

	year = int(fields[1])
	decade = year - year % 10 # year to decade conversion.

	return Row(twogram_first = first_column, twogram_last = last_column, decade = int(decade), occured = int(fields[2]))

# Map lines from sequence file to rows
rows = lines.map(dataConverter)

# Filter rows; drop terms that dont start with the word 'solar'
filtered_rows = rows.filter( lambda x: 'solar' == x['twogram_first'])

# Create data frame and table to query
ssql_df = spark.createDataFrame(filtered_rows).cache()

# Aggregate occurences of repeating solar terms within the same decade
ssql_df = ssql_df.groupby("decade", "twogram_first", "twogram_last").agg(ssql_f.sum("occured").alias('occur'))

# Window for filtering 20 most common solar terms per decade
window_20_common_terms_per_decade = window.Window.partitionBy(ssql_df['decade']).orderBy(ssql_df['occur'].desc())

# Window for filtering first occurence of any solar term across all decades
window_any_first_occurence = window.Window.partitionBy(ssql_df['twogram_last']).orderBy(ssql_df['decade'].asc()) 

# Select from data from using window functions the 20 most common solar terms and any solar term with first appearance.
# Arrange by decade in ascending order, occurence in descending order, 20 most common terms and any term appearance in ascending order of row numbers.
# Filter the dataframe allowing only the 20 most common solar terms and those terms just appearing for the first time ever.
ssql_dfmini = ssql_df.select('*', ssql_f.row_number().over(window_20_common_terms_per_decade).alias('twenty_common'), 
	ssql_f.row_number().over(window_any_first_occurence).alias('first_appearance')).\
	orderBy(ssql_df['decade'].asc(), ssql_df['occur'].desc(), ssql_f.col('twenty_common').asc(), ssql_f.col('first_appearance').asc()).\
	filter((ssql_f.col('twenty_common') <= 20) | (ssql_f.col('first_appearance') == 1))

# Comment out line block above and uncomment below  line block for allowing only the 20 most common solar term such that first appearance check is only for those 20 most common terms 
#ssql_dfmini = ssql_df.select('*', ssql_f.row_number().over(window_20_common_terms_per_decade).alias('twenty_common')).\
#	orderBy(ssql_df['decade'].asc(), ssql_df['occur'].desc(), ssql_f.col('twenty_common').asc()).\
#	filter((ssql_f.col('twenty_common') <= 20))

# From above selection merge the columns twogram_first and twogram_last of the 2-gram that were split previously in the mapping function into one column twogram.
# Group the outcome by decade then aggregate the terms from the twogram column into one row as a list per decade.
ssql_dfgroup = ssql_dfmini.select('*',ssql_f.concat_ws(' ',ssql_dfmini['twogram_first'],ssql_dfmini['twogram_last']).alias("twogram")).\
				groupby('decade').\
				agg(ssql_f.collect_list('twogram').alias("twogramlist"))

# Concatenate the list in each decade row from previous operations into a comma separated string in the same column.
# Order by decade.
ssql_dfgroup = ssql_dfgroup.withColumn('twogramlist', ssql_f.concat_ws(', ', ssql_dfgroup['twogramlist'])).\
				orderBy(ssql_dfgroup['decade'].asc())

# Coalesce the Spark dataframe into one partion the convert to a pandas dataframe for further processing.
twogram = ssql_dfgroup.coalesce(1).toPandas()
# Stop spark session to continue with simple operations about the data.
spark.stop()

# Can save data to a csv for viewing with another compatible software such as excel.
#twogram.to_csv("twogram.csv")

# Prepare PDF design for reporting as a HTML template as needed by weasyprint package.
html_top = '''
<html>
	<head>
		<title>Zola Electric 2Gram Challenge</title>
		<style>
			body{
				font-family: Times New Roman;
			}
			
			table{
				border:0;
				border-spacing:0;
				font-size:13px;
			}

			td{
				border:1px solid black;
				min-width: 150px;
				margin: 0;
				font-size:11px;
				text-align:left;
			}

			b{
				color:navy;
			}

			small{
				font-size:7px;
			}

			.decade{
				max-width:300px;
			}

			.terms{
				width:90%;
				max-width:700px;
			}
		</style>
	</head>
	<body>
		<center>
			<h2>Zola Electric 2Gram Challenge</h2>
			<h4>Prosper J. Mrutu</h4>
			</br>
			<table>
				<tr>
					<th class='decade'>DECADE</th>
					<th class='terms'>TERMS <small>(NB: first occurences highlighted in <b>bold navy blue</b> color)</small></th>
				</tr>
'''

html_bottom = '''
			</table>
		</center>
	</body>
</html>
'''

# Initialize html part for inserting rows in the final PDF report.
html_middle =''
# Initialize dictionary object for keeping check with first occurences so that it is possible to highlight these terms appropriately.
seen_befores = {}
# Initialize previous record variable for use in filling in missed decade gaps.
previous_decade = 0
# Loop the pandas data frame.
for ind, row in twogram.iterrows():

	this_decade = int(row['decade'])

	# Check if there are missing decade gaps and fill those with default NONE value.
	if previous_decade > 0 and this_decade - previous_decade > 10:
		decade_diff_mult = int((this_decade - previous_decade) / 10 - 1)
		for i in range(decade_diff_mult):
			html_middle += "<tr><td class='decade'>" + str(previous_decade + 10 * decade_diff_mult) + \
			"</td><td class='terms'>NONE</td></tr>"

	# Record previous decade processing continues in the loop.
	previous_decade = this_decade


	# Convert the comma separated list string of solar terms in this decade row into an actual sorted list object for better visual presentation and highlighting first appearances.
	twogramlisting = sorted(row['twogramlist'].split(","))

	# Start filling in the row to be presented in PDF.
	newtwogramlisting = "<td class='decade'>" + str(row['decade']) + "</td><td class='terms'>"
	appender = ''
	for i in range(len(twogramlisting)):
		twogramlistingistr = str(twogramlisting[i]).strip()

		# Create key for dictionary
		twogramlistingistrcompr = twogramlistingistr.replace(" ","")
		# with reverse for no repeatitions
		twogramlistingistrdict = twogramlistingistrcompr + twogramlistingistrcompr[::-1]

		# Check if the term has appeared before.
		if twogramlistingistrdict not in seen_befores:
			# If YES then highlight and record status.
			newtwogramlisting += appender + '<b>' + twogramlistingistr + '</b>'
			seen_befores[twogramlistingistrdict] = True
		else:
			# Otherwise just append to report.
			newtwogramlisting += appender + twogramlistingistr

		appender = ', '

	newtwogramlisting += '</td></tr>'

	html_middle += newtwogramlisting


# Store temporary HTML file use by weasyprint package for PDF creation.
htmlfile = '/tmp/pjmzolaelectric2gramchallengetemp.html'
# Where to store PDF file.
pdffile = 'pjmzolaelectric2gramchallenge.pdf'

# Create HTML temp file and use that with weasyprint package to create the final PDF file.
with open(htmlfile, 'w') as f:
	f.write(html_top + html_middle + html_bottom)
	f.close()

	wp.HTML(htmlfile).write_pdf(pdffile)

print("Saved.")
