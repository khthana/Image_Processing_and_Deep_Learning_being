while True :
    while True :
        try :
            base = float(input("Base : "))
        except :
                print("Error. The input has to be a number. try again")
        else :
           break

    while True :
        try :
            height = float(input("Height : "))
        except :
            print("Error. The input has to be a number. try again")
        else :
           break

    try :
        output = 1/2 * (base * height)
        print(output)
    except :
        print("Error. Somethings wrong,try again")
    else :
        break