/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.eclipse.emf.common.util.Enumerator;

/**
 * <!-- begin-user-doc -->
 * A representation of the literals of the enumeration '<em><b>Assignment Statement Enum</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see nqc.NqcPackage#getAssignmentStatementEnum()
 * @model
 * @generated
 */
public enum AssignmentStatementEnum implements Enumerator {
	/**
	 * The '<em><b>Assign</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ASSIGN_VALUE
	 * @generated
	 * @ordered
	 */
	ASSIGN(0, "assign", "assign"),

	/**
	 * The '<em><b>Addassign</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ADDASSIGN_VALUE
	 * @generated
	 * @ordered
	 */
	ADDASSIGN(1, "addassign", "addassign"),

	/**
	 * The '<em><b>Subassign</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SUBASSIGN_VALUE
	 * @generated
	 * @ordered
	 */
	SUBASSIGN(2, "subassign", "subassign"),

	/**
	 * The '<em><b>Mulassign</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #MULASSIGN_VALUE
	 * @generated
	 * @ordered
	 */
	MULASSIGN(3, "mulassign", "mulassign"),

	/**
	 * The '<em><b>Divassign</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #DIVASSIGN_VALUE
	 * @generated
	 * @ordered
	 */
	DIVASSIGN(4, "divassign", "divassign"),

	/**
	 * The '<em><b>Modassign</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #MODASSIGN_VALUE
	 * @generated
	 * @ordered
	 */
	MODASSIGN(5, "modassign", "modassign"),

	/**
	 * The '<em><b>Absassign</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #ABSASSIGN_VALUE
	 * @generated
	 * @ordered
	 */
	ABSASSIGN(6, "absassign", "absassign"),

	/**
	 * The '<em><b>Signassign</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #SIGNASSIGN_VALUE
	 * @generated
	 * @ordered
	 */
	SIGNASSIGN(7, "signassign", "signassign");

	/**
	 * The '<em><b>Assign</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Assign</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ASSIGN
	 * @model name="assign"
	 * @generated
	 * @ordered
	 */
	public static final int ASSIGN_VALUE = 0;

	/**
	 * The '<em><b>Addassign</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Addassign</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ADDASSIGN
	 * @model name="addassign"
	 * @generated
	 * @ordered
	 */
	public static final int ADDASSIGN_VALUE = 1;

	/**
	 * The '<em><b>Subassign</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Subassign</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SUBASSIGN
	 * @model name="subassign"
	 * @generated
	 * @ordered
	 */
	public static final int SUBASSIGN_VALUE = 2;

	/**
	 * The '<em><b>Mulassign</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Mulassign</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #MULASSIGN
	 * @model name="mulassign"
	 * @generated
	 * @ordered
	 */
	public static final int MULASSIGN_VALUE = 3;

	/**
	 * The '<em><b>Divassign</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Divassign</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #DIVASSIGN
	 * @model name="divassign"
	 * @generated
	 * @ordered
	 */
	public static final int DIVASSIGN_VALUE = 4;

	/**
	 * The '<em><b>Modassign</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Modassign</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #MODASSIGN
	 * @model name="modassign"
	 * @generated
	 * @ordered
	 */
	public static final int MODASSIGN_VALUE = 5;

	/**
	 * The '<em><b>Absassign</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Absassign</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #ABSASSIGN
	 * @model name="absassign"
	 * @generated
	 * @ordered
	 */
	public static final int ABSASSIGN_VALUE = 6;

	/**
	 * The '<em><b>Signassign</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>Signassign</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #SIGNASSIGN
	 * @model name="signassign"
	 * @generated
	 * @ordered
	 */
	public static final int SIGNASSIGN_VALUE = 7;

	/**
	 * An array of all the '<em><b>Assignment Statement Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final AssignmentStatementEnum[] VALUES_ARRAY =
		new AssignmentStatementEnum[] {
			ASSIGN,
			ADDASSIGN,
			SUBASSIGN,
			MULASSIGN,
			DIVASSIGN,
			MODASSIGN,
			ABSASSIGN,
			SIGNASSIGN,
		};

	/**
	 * A public read-only list of all the '<em><b>Assignment Statement Enum</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<AssignmentStatementEnum> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Assignment Statement Enum</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static AssignmentStatementEnum get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			AssignmentStatementEnum result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Assignment Statement Enum</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static AssignmentStatementEnum getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			AssignmentStatementEnum result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Assignment Statement Enum</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static AssignmentStatementEnum get(int value) {
		switch (value) {
			case ASSIGN_VALUE: return ASSIGN;
			case ADDASSIGN_VALUE: return ADDASSIGN;
			case SUBASSIGN_VALUE: return SUBASSIGN;
			case MULASSIGN_VALUE: return MULASSIGN;
			case DIVASSIGN_VALUE: return DIVASSIGN;
			case MODASSIGN_VALUE: return MODASSIGN;
			case ABSASSIGN_VALUE: return ABSASSIGN;
			case SIGNASSIGN_VALUE: return SIGNASSIGN;
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
	private AssignmentStatementEnum(int value, String name, String literal) {
		this.value = value;
		this.name = name;
		this.literal = literal;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getValue() {
	  return value;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
	  return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getLiteral() {
	  return literal;
	}

	/**
	 * Returns the literal value of the enumerator, which is its string representation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		return literal;
	}
	
} //AssignmentStatementEnum
