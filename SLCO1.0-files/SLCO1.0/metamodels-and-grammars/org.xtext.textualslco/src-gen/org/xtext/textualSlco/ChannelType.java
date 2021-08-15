/**
 */
package org.xtext.textualSlco;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.eclipse.emf.common.util.Enumerator;

/**
 * <!-- begin-user-doc -->
 * A representation of the literals of the enumeration '<em><b>Channel Type</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see org.xtext.textualSlco.TextualSlcoPackage#getChannelType()
 * @model
 * @generated
 */
public enum ChannelType implements Enumerator
{
  /**
   * The '<em><b>Synchronous</b></em>' literal object.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #SYNCHRONOUS_VALUE
   * @generated
   * @ordered
   */
  SYNCHRONOUS(0, "Synchronous", "sync"),

  /**
   * The '<em><b>Asynchronous Lossless</b></em>' literal object.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #ASYNCHRONOUS_LOSSLESS_VALUE
   * @generated
   * @ordered
   */
  ASYNCHRONOUS_LOSSLESS(1, "AsynchronousLossless", "async lossless"),

  /**
   * The '<em><b>Asynchronous Lossy</b></em>' literal object.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #ASYNCHRONOUS_LOSSY_VALUE
   * @generated
   * @ordered
   */
  ASYNCHRONOUS_LOSSY(2, "AsynchronousLossy", "async lossy");

  /**
   * The '<em><b>Synchronous</b></em>' literal value.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of '<em><b>Synchronous</b></em>' literal object isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @see #SYNCHRONOUS
   * @model name="Synchronous" literal="sync"
   * @generated
   * @ordered
   */
  public static final int SYNCHRONOUS_VALUE = 0;

  /**
   * The '<em><b>Asynchronous Lossless</b></em>' literal value.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of '<em><b>Asynchronous Lossless</b></em>' literal object isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @see #ASYNCHRONOUS_LOSSLESS
   * @model name="AsynchronousLossless" literal="async lossless"
   * @generated
   * @ordered
   */
  public static final int ASYNCHRONOUS_LOSSLESS_VALUE = 1;

  /**
   * The '<em><b>Asynchronous Lossy</b></em>' literal value.
   * <!-- begin-user-doc -->
   * <p>
   * If the meaning of '<em><b>Asynchronous Lossy</b></em>' literal object isn't clear,
   * there really should be more of a description here...
   * </p>
   * <!-- end-user-doc -->
   * @see #ASYNCHRONOUS_LOSSY
   * @model name="AsynchronousLossy" literal="async lossy"
   * @generated
   * @ordered
   */
  public static final int ASYNCHRONOUS_LOSSY_VALUE = 2;

  /**
   * An array of all the '<em><b>Channel Type</b></em>' enumerators.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private static final ChannelType[] VALUES_ARRAY =
    new ChannelType[]
    {
      SYNCHRONOUS,
      ASYNCHRONOUS_LOSSLESS,
      ASYNCHRONOUS_LOSSY,
    };

  /**
   * A public read-only list of all the '<em><b>Channel Type</b></em>' enumerators.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public static final List<ChannelType> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

  /**
   * Returns the '<em><b>Channel Type</b></em>' literal with the specified literal value.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public static ChannelType get(String literal)
  {
    for (int i = 0; i < VALUES_ARRAY.length; ++i)
    {
      ChannelType result = VALUES_ARRAY[i];
      if (result.toString().equals(literal))
      {
        return result;
      }
    }
    return null;
  }

  /**
   * Returns the '<em><b>Channel Type</b></em>' literal with the specified name.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public static ChannelType getByName(String name)
  {
    for (int i = 0; i < VALUES_ARRAY.length; ++i)
    {
      ChannelType result = VALUES_ARRAY[i];
      if (result.getName().equals(name))
      {
        return result;
      }
    }
    return null;
  }

  /**
   * Returns the '<em><b>Channel Type</b></em>' literal with the specified integer value.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public static ChannelType get(int value)
  {
    switch (value)
    {
      case SYNCHRONOUS_VALUE: return SYNCHRONOUS;
      case ASYNCHRONOUS_LOSSLESS_VALUE: return ASYNCHRONOUS_LOSSLESS;
      case ASYNCHRONOUS_LOSSY_VALUE: return ASYNCHRONOUS_LOSSY;
    }
    return null;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private final int value;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private final String name;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private final String literal;

  /**
   * Only this class can construct instances.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private ChannelType(int value, String name, String literal)
  {
    this.value = value;
    this.name = name;
    this.literal = literal;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public int getValue()
  {
    return value;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public String getName()
  {
    return name;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public String getLiteral()
  {
    return literal;
  }

  /**
   * Returns the literal value of the enumerator, which is its string representation.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public String toString()
  {
    return literal;
  }
  
} //ChannelType
