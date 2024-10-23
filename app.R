library(shiny)
library(shinydashboard)
library(ggplot2)
library(e1071)
library(caret)
library(shinyjs)
library(shinythemes)
library(DT)


#Generacion del data frame de 3000 registros divididos equitativamente en 3 clases
x=c(rnorm(1000,1000,100),rnorm(1000,2000,200),rnorm(1000,3000,400))
y=c(abs(rnorm(1000,50,25)),rnorm(1000,200,50),rnorm(1000,100,30))
clases=as.factor(c(rep(1,1000),rep(2,1000),rep(3,1000)))
datos=data.frame(x,y,clases)
w = nrow(datos)

ui <- dashboardPage(
  dashboardHeader(title = "SVM easyApp"),
  ## Sidebar content
  dashboardSidebar(
    sidebarMenu(
      menuItem("Datos", tabName = "datos", icon = icon("list")),
      menuItem("SVM", tabName = "svm", icon = icon("dashboard")),
      menuItem("Tune SVM", tabName = "tunesvm", icon = icon("th")),
      menuItem("Acerca de", tabName = "acerca", icon = icon("info-sign", lib="glyphicon")),
      menuItem("Código fuente", icon = icon("github"),
               href = "https://github.com/axelmora/svmdashboard/blob/master/svmdashboard.R"
      ),
      menuItem("LinkedIn", icon = icon("linkedin"),
               href = "https://linkedin.com/in/axelmora"
      ),
      menuItem("Twitter", icon = icon("twitter"),
               href = "https://twitter.com/axelmora93"
      )
    )
  ),
  ## Body content
  dashboardBody(
    tabItems(
      # First tab content
      tabItem(tabName = "svm",
              fluidRow(
                box(
                  title = "Controles",
                  status = "primary",
                  width = 3,
                           selectInput("kernel", "kernel",
                                       c(Lineal = "linear",
                                         Polinominal = "polynomial",
                                         Radial = "radial",
                                         Sigmoid = "sigmoid")),
                           uiOutput("ui")
                  ),
                tabBox(
                  width = 9,
                  tabPanel("Resumen SVM","Resumen",
                           verbatimTextOutput("sum")),
                  tabPanel("Grafica", "Grafica",
                           plotOutput("plot1")),
                  tabPanel("Matriz","Matriz de confusión",
                           verbatimTextOutput("pred"))
                )),
            fluidRow(
              valueBoxOutput("state"),
              valueBoxOutput("muestra"),
              valueBoxOutput("nsvm")
            )
      ),
      # Second tab content
      tabItem(tabName = "tunesvm",
              fluidRow(
                box(
                  title = "Controles",
                  status = "primary",
                  width = 3,
                     selectInput("kernel2", "kernel2",
                                       c(Lineal = "linear",
                                         Polinominal = "polynomial",
                                         Radial = "radial",
                                         Sigmoid = "sigmoid"))
                ),
                tabBox(
                  width = 9,
                  tabPanel("Resumen Tune SVM","Resumen",
                           verbatimTextOutput("sumTune")),
                  tabPanel("Grafica Tune SVM", "Grafica",
                           plotOutput("plot1T")),
                  tabPanel("Matriz Tune SVM","Matriz de confusión",
                           verbatimTextOutput("predT"))
                )),
              fluidRow(
                valueBoxOutput("stateT"),
                valueBoxOutput("muestraT"),
                valueBoxOutput("nsvmT")
              )
      ),
      tabItem(tabName = "datos",
              fluidRow(
              box(
                title = "Variables",
                status = "primary",
                width = 3,
                         checkboxInput("chek", label = "Usar archivo CSV", value = FALSE),
                         conditionalPanel("input.chek == false",
                                          sliderInput("numData", "Submuestra", value = 50, min = 10, max = w, step = 1)
                         ),
                         conditionalPanel("input.chek == true",
                                          fileInput('file1', 'Selecciona un CSV para analizar',
                                                    accept=c('text/csv', 
                                                             'text/comma-separated-values,text/plain', 
                                                             '.csv')),
                                          uiOutput("ui3")
                         ),
                         checkboxInput("var", label = "Analizar todas las variables", value = FALSE),
                         uiOutput("ui2"),
                         uiOutput("vd")
                  ),
              tabBox(
                width = 9,
                tabPanel("Entrenamiento",
                         DT::dataTableOutput("entrena")),
                tabPanel("Prueba",
                         DT::dataTableOutput("prueba")),
                tabPanel("Dataframe original",
                         DT::dataTableOutput("table")))
                )
              ),
      tabItem(tabName = "acerca",
              box(
                title = "Acerca de", status = "primary",
                width = 12,
                "SVM easyapp v 5.0: Aplicación web R Shiny de SVM", br(), br(),
                "Esta es una sencilla shiny app que implementa SVM (libreria 'e1071') en un dataset.
                Por defecto, trabaja con un data frame de 3000 observaciones 
                creados por R y clasificados en 3 clasess. 
                Puede optar por la opcion de usar un archivo CSV e importarlo en la aplicacion.
                Del total de datos, especifique una submuestra para el entrenamiento y especifique las
                variables x, y la discriminante en el panel Datos, donde además se muestra el datset original
                en una tabla. 
                En el panel SVM, seleccione el tipo de kernel y en base a eso configure los parametros necesarios. 
                Se muestra la tabla de confusion creada usando el paquete 'caret', el resumen del modelo, el 
                porcentaje correctamente clasificado segun la prediccion y el total de vectores soporte, 
                además de la gráfica del modelo creado.
                Puede consultar el modelo óptimo dado un kernel mediante la función best.tune en el panel Tune SVM.",
                br(),br(),
                "Basada en ejemplos y la documentacion de RStudio Shiny"
              )
      )
    )
  )
)

server <- function(input, output) {
  #Archivo CSV
  inFile <- reactive({input$file1}) 
  #Si no se ha ingresado un CSV, usa los datos preestablecidos
  dtPrev <- reactive({ 
    if (is.null(inFile())||input$chek == FALSE){
      datos
    }else{
      read.csv(inFile()$datapath)
    }})
  
  ### Parametros de la muestra y el tipo de kernel ###
  param <- reactiveValues() 
  observe(param$n <- if (input$chek == FALSE){input$numData}
          else{input$numData2})
  observe(param$k <- input$kernel)
  observe(param$c <- input$C)
  observe(param$g <- input$gamma)
  observe(param$cf <- input$coef0)
  observe(param$dg <- input$degree)
  
  ### Nombres de las columnas de la tabla de datos ###
  cols <- reactive(colnames(dtPrev())) 
  #se definen x, y, c que de las columnas seleccionadas de los datos 
  sc <- reactiveValues() 
  observe(sc$x <- which(cols() == input$xcol)) #variable x
  observe(sc$y <- which(cols() == input$ycol)) #variable y
  observe(sc$c <- which(cols() == input$ccol)) #variable que define las clases
  
  ### VARIABLES SELECCIONADAS ###
  pm <- reactiveValues()
  observe(pm$x <- as.character(input$xcol))
  observe(pm$y <- as.character(input$ycol))
  observe(pm$c <- as.character(input$ccol))
  
  ### FORMULAS ###
  for.svm1 <- reactive({as.formula(paste(pm$c,"~",pm$x,"+",pm$y))})
  for.svm2 <- reactive({as.formula(paste(pm$c,"~","."))})
  for.plot <- reactive({as.formula(paste(pm$y,"~",pm$x))})
  
  ### PREPARACION DATOS ###
  dtMaster <- reactive(dtPrev())
  
  indices <- function(data){
    set.seed(1234)
    sample(1:nrow(data),size=(param$n))
  }
  
  datax <- reactiveValues()
  observe(datax$indices <- indices(dtMaster()))
  observe(datax$entrenamiento <- dtMaster()[datax$indices,])
  observe(datax$test <- dtMaster()[-datax$indices,])
  
  ### SVM ###
  svmx <- reactive({
    if(input$var == FALSE){
      switch(input$kernel,
             "linear" =  svm(for.svm1(), data=datax$entrenamiento, 
                             kernel=param$k, cost = param$c),
             
             "polynomial" =  svm(for.svm1(), data=datax$entrenamiento, 
                                 kernel=param$k, cost = param$c, gamma = param$g, coef0 = param$cf, degree = param$dg),
             
             "radial" = svm(for.svm1(), data=datax$entrenamiento, 
                            kernel=param$k, cost = param$c, gamma = param$g),
             
             "sigmoid" = svm(for.svm1(), data=datax$entrenamiento, 
                             kernel=param$k, cost = param$c, gamma = param$g, coef0 = param$cf))
    }else if (input$var == TRUE){
      switch(input$kernel,
             "linear" =  svm(for.svm2(), data=datax$entrenamiento, 
                             kernel=param$k, cost = param$c),
             
             "polynomial" =  svm(for.svm2(), data=datax$entrenamiento, 
                                 kernel=param$k, cost = param$c, gamma = param$g, coef0 = param$cf, degree = param$dg),
             
             "radial" = svm(for.svm2(), data=datax$entrenamiento, 
                            kernel=param$k, cost = param$c, gamma = param$g),
             
             "sigmoid" = svm(for.svm2(), data=datax$entrenamiento, 
                             kernel=param$k, cost = param$c, gamma = param$g, coef0 = param$cf))
    }
  })
  
  ### BEST TUNE ###
  svmx.tune <- reactive({
    if(input$var == FALSE){
      best.tune(svm,for.svm1(), data = datax$entrenamiento, kernel = input$kernel2)
    }else if (input$var == TRUE){
      best.tune(svm,for.svm2(), data = datax$entrenamiento, kernel = input$kernel2)
    }
  })
  
  ### UI DINAMICAS ###
  output$ui <- renderUI({
    if (is.null(input$kernel))
      return()
    
    switch(input$kernel,
           
           "linear" = tabPanel("Lineal", 
                               numericInput('C','Training parameter C', value = 1)),
           
           "polynomial" = tabPanel("Polinominal",
                                   numericInput('C','Training parameter C', value = 1),
                                   numericInput('gamma','Training parameter gamma', value = 0.25),
                                   numericInput('coef0','Training parameter coef0', value = 1),
                                   numericInput('degree','Training parameter degree', value = 3)),
           "radial" = tabPanel("Radial",
                               numericInput('C','Training parameter C', value = 1),
                               numericInput('gamma','Training parameter gamma', value = 0.25)),
           "sigmoid" = tabPanel("Sigmoid",
                                numericInput('C','Training parameter C', value = 1),
                                numericInput('gamma','Training parameter gamma', value = 0.25),
                                numericInput('coef0','Training parameter coef0', value = 1))
    )
  })
  
  output$ui2 <- renderUI({
    tabPanel("Columnas",
             selectInput("xcol", "Variable x", cols(), selected = cols()[1]),
             selectInput("ycol", "Variable y", cols(), selected = cols()[2]))
  })
  
  output$ui3 <- renderUI({
    sliderInput("numData2", "Submuestra", value = 50, min = 10, max = nrow(dtPrev()), step = 1)
  })
  
  output$vd <- renderUI({
    tabPanel("VD",
             selectInput("ccol", "Variable discriminadora", cols(), selected = cols()[3]))
  })
  
  ### GRAFICAS ###
  output$plot1 <- renderPlot({
    plot(svmx(), datax$entrenamiento, for.plot())
  })
  
  output$plot1T <- renderPlot({
    plot(svmx.tune(), datax$entrenamiento, for.plot())
  })
  
  ### MATRIZ DE CONFUSION ###
  output$pred <- renderPrint({
    #Prediccion de los restantes
    asignado <- predict(svmx(),new=datax$test)
    #Tabla de confusion
    #mc <- with(datax$test,(pred=table(asignado,datax$test[,sc$c])))
    mc <- confusionMatrix(asignado, datax$test[,sc$c])
    mc
  })
  
  output$predT <- renderPrint({
    #Prediccion de los restantes
    asignadoT <- predict(svmx.tune(),new=datax$test)
    #Tabla de confusion
    #mcT <- with(datax$test,(pred=table(asignadoT,datax$test[,sc$c])))
    mcT <- confusionMatrix(asignadoT, datax$test[,sc$c])
    mcT
  })
  
  ### SUMARIOS SVM Y TUNE ###
  output$sum <- renderPrint({
    summary(svmx())
  })
  
  output$sumTune <- renderPrint({
    summary(svmx.tune())
  })
  
  ### PORCENTAJE CLASIFICADO ##
  output$state <- renderValueBox({
    #Prediccion de los restantes
    asignado <- predict(svmx(),new=datax$test)
    #Tabla de confusion
    mc <- with(datax$test,(pred=table(asignado,datax$test[,sc$c])))
    #porcentaje correctamente clasificados
    if(is.nan(correctos <- sum(diag(mc)) / nrow(datax$test) *100)){
      correctos <- 0
    }else {
      correctos <- sum(diag(mc)) / nrow(datax$test) *100
    }
    valueBox(paste0(round(correctos,3),"%"), "Clasificacion correcta", 
             icon=icon("thumbs-up",lib = "glyphicon"), color = "yellow")
  })
  
  output$muestra <-renderValueBox({
    tot <- nrow(dtPrev())
    mst <- param$n
    pct <- (mst/tot)*100
    valueBox(paste0(round(pct,3),"%"),"Muestra de entrenamiento",
             icon=icon("list"), color = "blue")
  })
  
  output$muestraT <-renderValueBox({
    tot <- nrow(dtPrev())
    mst <- param$n
    pct <- (mst/tot)*100
    valueBox(paste0(round(pct,3),"%"),"Muestra de entrenamiento",
             icon=icon("list"), color = "blue")
  })
  
  output$stateT <- renderValueBox({
    #Prediccion de los restantes
    asignadoT <- predict(svmx.tune(),new=datax$test)
    #Tabla de confusion
    mcT <- with(datax$test,(pred=table(asignadoT,datax$test[,sc$c])))
    #porcentaje correctamente clasificados
    if(is.nan(correctosT <- sum(diag(mcT)) / nrow(datax$test) *100)){
      correctosT <- 0
    }else {
      correctosT <- sum(diag(mcT)) / nrow(datax$test) *100
    }
    valueBox(paste0(round(correctosT,3),"%"), "Clasificacion correcta", 
             icon=icon("thumbs-up",lib = "glyphicon"), color = "yellow")
  })
  
  output$nsvm <- renderValueBox({
    n <- svmx()$tot.nSV
    valueBox(n, "Total de SV", icon = icon("ok-sign", lib = "glyphicon"))
  })
  
  output$nsvmT <- renderValueBox({
    nT <- svmx.tune()$tot.nSV
    valueBox(nT, "Total de SV", icon = icon("ok-sign", lib = "glyphicon"))
  })
  
  ### TABLAS ###
  output$entrena <- DT::renderDataTable({
    DT::datatable(datax$entrenamiento, options=list(orderClasses = TRUE))
  })
  
  output$prueba <- DT::renderDataTable({
    DT::datatable(datax$test, options=list(orderClasses = TRUE))
  })
  
  output$table <- DT::renderDataTable({
    DT::datatable(dtMaster(), options=list(orderClasses = TRUE))
  })
}

shinyApp(ui, server)
